from typing import List, Dict

def parse_predictions(path: str) -> List[Dict]:
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        sent_id = -1
        for line in f:
            line=line.strip()
            if line.startswith('SENT'):
                parts = line.split(':',1)
                try:
                    sent_id = int(parts[0].split()[1])
                except Exception:
                    sent_id = -1
                continue
            if line.startswith('TUP') and ':' in line:
                tup = line.split(':',1)[1].strip()
                if tup.startswith('(') and tup.endswith(')'):
                    tup = tup[1:-1]
                parts = [p.strip() for p in tup.split(';')]
                if len(parts) >= 3:
                    triples.append({
                        "sent_id": sent_id,
                        "subject": parts[0],
                        "predicate": parts[1],
                        "object": parts[2]
                    })
    return triples
