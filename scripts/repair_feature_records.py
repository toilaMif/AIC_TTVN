from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg
from retrieval.config import settings
from retrieval.indexes.milvus import SELF_AICV3_COLLECTION, stable_milvus_pk

url = settings.database_url.replace('+psycopg','')
with psycopg.connect(url) as conn, conn.cursor() as cur:
    cur.execute('TRUNCATE feature_records, feature_jobs RESTART IDENTITY CASCADE')
    cur.execute('SELECT video_id, keyframe_id FROM keyframes ORDER BY video_id, n')
    rows = cur.fetchall()
    groups = {}
    for video_id, keyframe_id in rows: groups.setdefault(video_id, []).append(keyframe_id)
    for video_id, ids in groups.items():
        version = f'ViT-B-32/laion2b_s34b_b79k:aicv3:{video_id}'
        job = f'visual-aicv3-{video_id.lower()}'
        cur.execute('''INSERT INTO feature_jobs
          (job_id,model_name,model_version,embedding_version,dimension,dtype,expected_count,completed_count,status)
          VALUES (%s,%s,%s,%s,512,%s,%s,%s,%s)''',
          (job,'ViT-B-32/laion2b_s34b_b79k','aicv3',version,'float32',len(ids),len(ids),'completed'))
        for kf in ids:
            cur.execute('''INSERT INTO feature_records
              (job_id,keyframe_id,embedding_version,milvus_collection,milvus_pk,import_status)
              VALUES (%s,%s,%s,%s,%s,%s)''',
              (job,kf,version,SELF_AICV3_COLLECTION,stable_milvus_pk(kf),'imported'))
print(f'repaired videos={len(groups)} keyframes={len(rows)}')
