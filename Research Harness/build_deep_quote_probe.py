"""Build a standalone V24-derived Gas/Kit quote-demand probe, not a final trader."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TAIL=r'''

# RESEARCH PROBE: tests quote distance, with full-session aggregate telemetry.
# Arms: 0 unchanged V24; 1 one cent wider; 2 three cents wider; 3 one cent tighter.
# Probe only ticks 100..299 of each 1000-tick block to reduce practice-score cost.
# Counter columns: eligible ticks, submitted units, fill units, entry-edge cents,
# 20-tick markout cents, markout units, absolute inventory sum.
_DemandProbeBase = Trader

class Trader(_DemandProbeBase):
    def run(self,state):
        saved=self._load(state.trader_data)
        probe=saved.get('demand_probe',{'counts':{},'pending':{},'marks':[]})
        tick=state.timestamp//100
        def counter(tag,arm):
            key=tag+str(arm)
            if key not in probe['counts']:probe['counts'][key]=[0,0,0,0.,0.,0,0]
            return probe['counts'][key]
        future=[]
        for due,symbol,tag,arm,side,quantity,price in probe['marks']:
            mid=self._mid(state.order_depths[symbol])
            if tick>=due and mid is not None:
                c=counter(tag,arm);c[4]+=side*quantity*(mid-price);c[5]+=quantity
            else:future.append([due,symbol,tag,arm,side,quantity,price])
        probe['marks']=future
        for symbol,info in probe['pending'].items():
            tag,arm,mid,submitted=info
            for trade in state.own_trades.get(symbol,[]):
                side=1 if trade.buyer=='SUBMISSION' else -1
                # Only fills matching our changed passive price/side enter probe totals.
                if [side,trade.price] not in submitted:continue
                c=counter(tag,arm);c[2]+=trade.quantity;c[3]+=side*trade.quantity*(mid-trade.price)
                probe['marks'].append([tick+19,symbol,tag,arm,side,trade.quantity,trade.price])
        probe['pending']={}
        orders,conv,data=super().run(state)
        stats=self._load(data)
        # Deterministic scrambled 25-tick blocks; no future prices used.
        block=tick//25
        arm=((block*2654435761)&0xffffffff)>>30
        offset=(0,-1,-3,1)[arm]
        for symbol,tag in ((MASKS,'M'),(KIT,'K')):
            if not 100 <= tick%1000 < 300:continue
            quotes=orders.get(symbol,[])
            depth=state.order_depths.get(symbol)
            if not depth or not quotes:continue
            bid,ask=depth.best_bid(),depth.best_ask()
            if bid is None or ask is None:continue
            if any((o.quantity>0 and o.price>=ask) or (o.quantity<0 and o.price<=bid) for o in quotes):continue
            if symbol==KIT and stats.get('kit_mm_pending') is None:continue
            c=counter(tag,arm);c[0]+=1;c[6]+=abs(state.position.get(symbol,0))
            submitted=[]
            replacement=[]
            for o in quotes:
                side=1 if o.quantity>0 else -1
                price=o.price+side*offset
                price=min(price,ask-1) if side>0 else max(price,bid+1)
                replacement.append(Order(symbol,int(price),int(o.quantity)))
                submitted.append([side,int(price)])
                c[1]+=abs(o.quantity)
            orders[symbol]=replacement
            probe['pending'][symbol]=[tag,int(arm),(bid+ask)/2,submitted]
        if tick%1000==999:
            for tag in ('M','K'):
                rows=[[round(v,2) for v in counter(tag,a)] for a in range(4)]
                print('DEMAND|'+str(tick)+'|'+tag+'|'+json.dumps(rows,separators=(',',':')))
        stats['demand_probe']=probe
        return orders,conv,self._save(stats)
'''

def main():
    path=ROOT/'research/deep_quote_probe_submission.py'
    path.write_text((ROOT/'trader_v24.py').read_text()+TAIL,encoding='utf-8')
    print(path, path.stat().st_size)

if __name__=='__main__':main()
