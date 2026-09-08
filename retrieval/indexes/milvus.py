"""Milvus collection lifecycle helpers."""

from pymilvus import DataType, MilvusClient

DEFAULT_COLLECTION = "keyframe_visual_vit_b32_v1"
SELF_AICV3_COLLECTION = "keyframe_visual_openclip_vit_b32_aicv3"
CAPTION_TEXT_COLLECTION = "frame_caption_text_viembed_v1"


def stable_milvus_pk(keyframe_id: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(keyframe_id.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def create_visual_collection(
    uri: str, collection_name: str = DEFAULT_COLLECTION, dimension: int = 512
) -> bool:
    client = MilvusClient(uri=uri)
    if client.has_collection(collection_name):
        return False
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="keyframe_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="frame_idx", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    return True


def upsert_visual_vectors(
    uri: str, collection_name: str, records: list[dict[str, object]], dimension: int
) -> int:
    create_visual_collection(uri, collection_name, dimension)
    client = MilvusClient(uri=uri)
    primary_keys = [int(record["pk"]) for record in records]
    existing = client.get(
        collection_name=collection_name,
        ids=primary_keys,
        output_fields=["pk"],
    )
    existing_keys = {int(record["pk"]) for record in existing}
    missing = [record for record in records if int(record["pk"]) not in existing_keys]
    if missing:
        client.insert(collection_name=collection_name, data=missing)
    client.flush(collection_name=collection_name)
    return len(missing)


def search_visual_vector(
    uri: str, collection_name: str, vector: list[float], limit: int = 5
) -> list[dict[str, object]]:
    client = MilvusClient(uri=uri)
    results = client.search(
        collection_name=collection_name,
        data=[vector],
        anns_field="embedding",
        limit=limit,
        output_fields=["keyframe_id", "video_id", "frame_idx"],
        search_params={"metric_type": "IP", "params": {}},
    )
    return list(results[0])
