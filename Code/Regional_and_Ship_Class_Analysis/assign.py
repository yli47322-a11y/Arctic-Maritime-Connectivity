import json
import struct
from collections import Counter

import numpy as np
from numba import njit
from analyze import HERE, read

# Global LME 66 data incorporates the revised 2013 Arctic polygons.
SEAS = {
    60:'法罗群岛', 59:'冰岛陆架', 19:'格陵兰海—东格陵兰', 21:'挪威海',
    20:'巴伦支海', 58:'喀拉海', 57:'拉普捷夫海', 56:'东西伯利亚海',
    1:'东白令海', 65:'阿留申群岛', 53:'西白令海', 54:'北白令—楚科奇海',
    55:'波弗特海', 66:'加拿大高北极—北格陵兰', 18:'巴芬湾—西格陵兰',
    63:'哈得孙湾', 9:'拉布拉多—纽芬兰',
    2:'阿拉斯加湾', 52:'鄂霍次克海', 22:'北海', 23:'波罗的海',
}


def project(xy):
    lon, lat = np.deg2rad(xy.T)
    rho = 2*6371.0088*np.tan((np.pi/2-lat)/2)
    return np.column_stack((rho*np.sin(lon), -rho*np.cos(lon)))


@njit
def distances(points, a, b):
    result = np.empty(len(points))
    x0, y0, x1, y1 = a[:,0], a[:,1], b[:,0], b[:,1]
    dx, dy = x1-x0, y1-y0
    for i in range(len(points)):
        x, y = points[i]
        best = np.inf
        inside = False
        for j in range(len(a)):
            px, py = x-x0[j], y-y0[j]
            if (y0[j]>y) != (y1[j]>y) and px < dx[j]*py/dy[j]:
                inside = not inside
            norm = dx[j]*dx[j]+dy[j]*dy[j]
            t = min(1., max(0., (px*dx[j]+py*dy[j])/norm)) if norm > 0 else 0.
            dist = (px-t*dx[j])**2+(py-t*dy[j])**2
            if dist < best:
                best = dist
        result[i] = 0. if inside else np.sqrt(best)
    return result


def main():
    shp = (HERE/'lme66.shp').read_bytes()
    dbf = (HERE/'lme66.dbf').read_bytes()
    n, head, width = struct.unpack_from('<IHH',dbf,4)
    features = {}
    pos = 100
    for i in range(n):
        _, nwords = struct.unpack_from('>ii',shp,pos)
        pos += 8
        number = int(float(dbf[head+i*width+10:head+i*width+29]))
        if number in SEAS:
            parts, pts = struct.unpack_from('<ii',shp,pos+36)
            starts = np.r_[np.frombuffer(shp,dtype='<i4',count=parts,offset=pos+44),pts]
            xy = np.frombuffer(shp,dtype='<f8',count=pts*2,offset=pos+44+4*parts).reshape(-1,2)
            rings = [project(xy[a:b]) for a,b in zip(starts[:-1],starts[1:])]
            features[number] = (np.concatenate([r[:-1] for r in rings]),np.concatenate([r[1:] for r in rings]))
        pos += nwords*2
    records = read('聚居点与出发格对应.csv')
    coords = project(np.array([(float(r['settlement_longitude']),float(r['settlement_latitude'])) for r in records]))
    ids = list(SEAS)
    ds = np.column_stack([distances(coords,*features[k]) for k in ids])
    choices = ds.argmin(1)
    actual = sorted(set(choices))
    active = {j:i for i,j in enumerate(actual)}
    mapping = {r['entity_id']:active[j] for r,j in zip(records,choices)}
    regions = [SEAS[ids[j]] for j in actual]
    summary = [{'name':regions[i],'lme':ids[j],'n':int((choices==j).sum()),'countries':dict(Counter(r['country_code'] for r,k in zip(records,choices) if k==j)),
                'fallback':int(((choices==j)&(ds.min(1)>0)).sum())} for i,j in enumerate(actual)]
    out = {'regions':regions,'assignments':mapping,'summary':summary,'nearest_count':int((ds.min(1)>0).sum()),'max_km':float(ds.min(1).max())}
    (HERE/'assignment.json').write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True),flush=True)
    print('nearest',out['nearest_count'],'max km',out['max_km'],flush=True)


if __name__ == '__main__':
    main()
