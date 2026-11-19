"""Unpack a locally downloaded case 4 dev bundle. Does not access the network."""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def manifest_entry(path):
    content = path.read_bytes().replace(b'\r\n', b'\n')
    return {
        'path': path.relative_to(ROOT).as_posix(),
        'bytes': len(content),
        'sha256': hashlib.sha256(content).hexdigest(),
    }


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Unsafe identifier')
    return value


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((ROOT / 'raw' / 'case4-dev-v2.json').read_text(encoding='utf-8-sig'))
    dataset = bundle['dataset']
    articles, experts = dataset['articles'], dataset['experts']
    if len(articles) != 20 or len(experts) != 3:
        raise ValueError('Expected 20 articles and 3 experts')
    if len({a['article_id'] for a in articles}) != 20:
        raise ValueError('Duplicate article identifiers')
    sections = []
    for article in articles:
        aid = safe_id(article['article_id'])
        write_json(ROOT / 'articles' / f'{aid}.json', article)
        lic = article['license']
        lines = [f"# {article['title']}", '', f"Article ID: `{aid}`", '',
                 'Authors: ' + ', '.join(article['authors']), '', f"Year: {article['year']}", '',
                 f"DOI: https://doi.org/{article['doi']}", '',
                 f"License: {lic['name']} {lic['version']} — {lic['url']}", '']
        for section in article['sections']:
            if not section['text'].strip():
                raise ValueError('Empty source section')
            lines += [f"## {section['title']}", '', f"Section ID: `{section['section_id']}`", '', section['text'], '']
            sections.append({'article_id': aid, 'article_title': article['title'], **section,
                             'doi': article['doi'], 'license': lic})
        (ROOT / 'articles' / f'{aid}.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(ROOT / 'articles.json', articles)
    write_json(ROOT / 'experts.json', experts)
    combined = ['# Эксперты из набора организаторов', '']
    for expert in experts:
        eid = safe_id(expert['expert_id'])
        write_json(ROOT / 'experts' / f'{eid}.json', expert)
        lines = [f"# {expert['role']}", '', f"Expert ID: `{eid}`", '', '## Стиль речи', '', expert['speech_style'], '']
        (ROOT / 'experts' / f'{eid}.md').write_text('\n'.join(lines), encoding='utf-8')
        combined += lines
    (ROOT / 'experts.md').write_text('\n'.join(combined), encoding='utf-8')
    (ROOT / 'sections.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in sections), encoding='utf-8')
    write_json(ROOT / 'judge_prompts.json', dataset.get('judge_prompts', []))
    write_json(ROOT / 'dataset_meta.json', dataset['meta'])
    write_json(ROOT / 'dataset.json', dataset)
    write_json(ROOT / 'reference.json', bundle.get('reference'))
    for name in ('users', 'chat_history', 'requests'):
        write_json(ROOT / f'{name}.json', dataset[name])
    for prompt in dataset.get('judge_prompts', []):
        pid = safe_id(prompt['prompt_id'])
        write_json(ROOT / 'judge_prompts' / f'{pid}.json', prompt)
        (ROOT / 'judge_prompts' / f'{pid}.md').write_text(
            f"# {prompt['purpose']}\n\nID: `{pid}`\n\nVersion: {prompt['version']}\n\n"
            f"Model version: {prompt['model_version']}\n\nTemperature: {prompt['temperature']}\n\n{prompt['text']}\n",
            encoding='utf-8')
    spec = json.loads((ROOT / 'openapi.json').read_text(encoding='utf-8-sig'))
    for name, schema in spec['components']['schemas'].items():
        if name.startswith('Case4') or name in ('MedcheckResources', 'MedcheckRequest', 'MedcheckResponse',
                                               'MedicalVerdict', 'ValidationResult', 'Error'):
            write_json(ROOT / 'schemas' / f'{name}.schema.json', schema)
    write_json(ROOT / 'case4_endpoints.json', {p: v for p, v in spec['paths'].items() if p.startswith('/case/4/')})
    index = ['# Статьи из набора организаторов', '', '| ID | Название | Год | Разделов |', '| --- | --- | --- | --- |']
    for article in articles:
        aid = article['article_id']
        title = article['title'].replace('|', '\\|')
        index.append(f"| [{aid}](articles/{aid}.md) | {title} | {article['year']} | {len(article['sections'])} |")
    (ROOT / 'ARTICLES_INDEX.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    counts = {k: len(dataset[k]) for k in ('articles', 'experts', 'users', 'chat_history', 'requests', 'judge_prompts')}
    reference = bundle.get('reference')
    counts['reference_requests'] = len(reference.get('requests', [])) if isinstance(reference, dict) else 0
    write_json(ROOT / 'inventory.json', {'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                'meta': dataset['meta'], 'counts': counts, 'sections': len(sections)})
    del bundle, dataset
    generated = {'manifest.json', 'validation_report.json', 'api-health.json'}
    files = sorted(p for p in ROOT.rglob('*')
                   if p.is_file() and p.name not in generated and '__pycache__' not in p.parts)
    write_json(ROOT / 'manifest.json', {'articles': 20, 'experts': 3, 'sections': len(sections),
               'counts': counts, 'full_bundle_retained': True,
               'files': [manifest_entry(p) for p in files]})
    print(f'Saved 20 articles, 3 experts, {len(sections)} sections; full development bundle retained locally')
    print('Inventory counts:', counts)
