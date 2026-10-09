"""Publish aggregate measurements only; benchmark items and host paths stay local."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'data/external_evaluation/results'
SOURCES = {
    'horscht-v0.1': ('new-data-last-block-fast-2026-10-09', 'last-block'),
    'previous-last-block': ('current-corpus-last-block-2026-10-05', 'last-block'),
    'original-qwen3-4b-immediate-answer': ('original-qwen-label-likelihood-2026-10-04', None),
}


def main():
    output = ROOT / 'release/evidence'
    output.mkdir(parents=True, exist_ok=True)
    for name, (folder, candidate) in SOURCES.items():
        path = RESULTS / folder / 'blind-aggregate-result.json'
        source = json.loads(path.read_text())
        result = source['candidate_results'][candidate] if candidate else source
        public = {
            'schema_version': 1,
            'model': name,
            'source_aggregate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'Exploratory reuse of the same 17,874-question external suite; aggregate metrics only.',
            'overall': result['overall'],
            'by_source_slice': result['by_source_slice'],
        }
        assert public['overall']['examples'] == 17874
        assert sum(row['correct'] for row in public['by_source_slice'].values()) == public['overall']['correct']
        (output / f'{name}.json').write_text(json.dumps(public, indent=2) + '\n')
    print('Exported three aggregate-only evidence files.')


if __name__ == '__main__':
    main()
