from pathlib import Path
import osmnx as ox, networkx as nx
BASE=Path(__file__).resolve().parent
GRAPH=BASE/'data/processed/safepath_ml_safety_graph.graphml'
MAX_DETOUR=10.0; MAX_CANDIDATES=100

def load():
    G=ox.load_graphml(GRAPH)
    for _,_,_,d in G.edges(keys=True,data=True):
        d['length']=float(d.get('length',1)); d['safety_score']=float(d.get('safety_score',50)); d['safety_risk']=float(d.get('safety_risk',max(0,100-d['safety_score'])))
    return G

def simple(G):
    U=nx.Graph()
    for u,v,d in ox.convert.to_undirected(G).edges(data=True):
        if not U.has_edge(u,v) or float(d['length'])<float(U[u][v]['length']): U.add_edge(u,v,**d)
    return U

def stats(U,r):
    dist=risk=0.; scores=[]
    for u,v in zip(r[:-1],r[1:]):
        d=U[u][v]; dist+=float(d['length']); risk+=float(d['safety_risk']); scores.append(float(d['safety_score']))
    return dist,risk,sum(scores)/len(scores)

def pair(U):
    ns=list(U.nodes)
    for s in ns:
        for e in ns:
            if s==e: continue
            try:
                g=nx.shortest_simple_paths(U,s,e,weight='length')
                for i,_ in enumerate(g):
                    if i>=2: return s,e
            except nx.NetworkXNoPath: pass
    raise RuntimeError('No node pair with multiple routes found.')

def safest(U,s,e,fast_dist):
    limit=fast_dist*(1+MAX_DETOUR/100); best=None; key=None
    try:
        g=nx.shortest_simple_paths(U,s,e,weight='safety_risk')
        for i,r in enumerate(g):
            if i>=MAX_CANDIDATES: break
            d,risk,score=stats(U,r)
            if d<=limit and (key is None or (risk,d)<key): best=r; key=(risk,d)
    except nx.NetworkXNoPath: pass
    return best or nx.shortest_path(U,s,e,weight='length')

def main():
    G=load(); U=simple(G); s,e=pair(U); fast=nx.shortest_path(U,s,e,weight='length'); fd,fr,fs=stats(U,fast); safe=safest(U,s,e,fd); sd,sr,ss=stats(U,safe)
    extra=(sd-fd)/fd*100 if fd else 0; change=ss-fs; reduction=fr-sr
    print('Safety graph loaded!'); print(f'Nodes: {len(G.nodes)}'); print(f'Edges: {len(G.edges)}'); print('\nRouting graph created!'); print(f'Nodes: {len(U.nodes)}'); print(f'Edges: {len(U.edges)}'); print('\n================================\nROUTE PAIR FOUND\n================================'); print('Start node:',s); print('Destination node:',e)
    print('\n================================\nFASTEST ROUTE\n================================'); print('Road segments:',len(fast)-1); print('Distance:',round(fd,2),'metres'); print('Average safety score:',round(fs,2)); print('Total risk:',round(fr,2))
    print('\n================================\nML SAFETY-AWARE ROUTE\n================================'); print('Road segments:',len(safe)-1); print('Distance:',round(sd,2),'metres'); print('Average safety score:',round(ss,2)); print('Total risk:',round(sr,2))
    print('\n================================\nROUTE COMPARISON\n================================'); print('Maximum allowed detour:',MAX_DETOUR,'%'); print('Actual extra distance:',round(extra,2),'%'); print('Safety improvement:',round(change,2),'points'); print('Risk reduction:',round(reduction,2)); print('\nFastest route:'); print(fast); print('\nML safety-aware route:'); print(safe); print('\n================================\nSAFEPATH DECISION\n================================'); print('Recommended: ML SAFETY-AWARE ROUTE' if change>0 else 'Recommended: FASTEST ROUTE'); print('Reason: higher predicted safety within the allowed travel detour.' if change>0 else 'Reason: ML safety-aware route does not improve predicted safety within the allowed travel detour.')
if __name__=='__main__': main()
