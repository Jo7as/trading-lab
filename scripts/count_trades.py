import json
closed=0; total=0; open_t=0
with open('state/trades.jsonl') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        total+=1
        try: t=json.loads(line)
        except: continue
        s=t.get('status')
        if s=='closed': closed+=1
        elif s=='open': open_t+=1
print(f'total={total} closed={closed} open={open_t}')
