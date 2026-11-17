"""Verify local source extraction without network calls or displaying dataset contents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


bundle = read('raw/case4-dev-v2.json')
dataset = bundle['dataset']
assert read('dataset.json') == dataset
assert read('reference.json') == bundle['reference']
assert read('articles.json') == dataset['articles']
assert read('experts.json') == dataset['experts']
assert len(dataset['articles']) == 20
assert len(dataset['experts']) == 3
assert {e['expert_id'] for e in dataset['experts']} == {'clinician', 'biohacker', 'nutritionist'}
assert len(list((ROOT / 'articles').glob('*.json'))) == 20
assert len(list((ROOT / 'articles').glob('*.md'))) == 20
assert len(list((ROOT / 'experts').glob('*.json'))) == 3
assert len(list((ROOT / 'experts').glob('*.md'))) == 3
citations = set()
for article in dataset['articles']:
    assert article['license']['name'] == 'CC BY'
    assert article['authors'] and article['doi'].startswith('10.')
    assert read('articles/' + article['article_id'] + '.json') == article
    for section in article['sections']:
        assert section['text'].strip()
        key = article['article_id'], section['section_id']
        assert key not in citations
        citations.add(key)
sections = [json.loads(line) for line in (ROOT / 'sections.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(sections) == len(citations)
assert {(s['article_id'], s['section_id']) for s in sections} == citations
requests = dataset['requests']
assert len({r['record_id'] for r in requests}) == len(requests)
reference = bundle['reference']
assert isinstance(reference, dict)
assert {r['record_id'] for r in reference['requests']} == {r['record_id'] for r in requests}

# All manifest paths stay in data; compare byte-for-byte hashes of all listed files.
manifest = read('manifest.json')
for item in manifest['files']:
    path = (ROOT / item['path']).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert path.stat().st_size == item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']

report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'articles': 20, 'experts': 3, 'sections': len(sections), 'requests': len(requests),
          'reference_requests': len(reference['requests']), 'manifest_files_checked': len(manifest['files']),
          'checks': ['full bundle retained', 'extracted data matches original', 'article and expert counts',
                     'non-empty sections', 'CC BY attribution retained', 'section ID uniqueness',
                     'request/reference ID coverage', 'file sizes and SHA256']}
(ROOT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Verification passed:', report['articles'], 'articles,', report['experts'], 'experts,', len(sections), 'sections')
