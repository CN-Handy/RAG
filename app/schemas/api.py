from pydantic import BaseModel, Field
from typing import Union, List, Tuple, Dict, Optional


class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    token: str
    model: str


class EmbeddingResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    vector: List[List[float]] = Field(description="文本对应的向量表示")
    response_code: int = Field(description="响应代码")
    response_msg: str = Field(description="响应信息")
    process_status: str = Field(description="处理状态")
    processing_time: float = Field(description="处理耗时（秒）")


class RerankRequest(BaseModel):
    text_pair: List[Tuple[str, str]]
    token: str
    model: str


class RerankResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    vector: List[float]
    response_code: int = Field(description="响应代码")
    response_msg: str = Field(description="响应信息")
    process_status: str = Field(description="处理状态")
    processing_time: float = Field(description="处理耗时（秒）")


class KnowledgeRequest(BaseModel):
    category: str
    title: str


class KnowledgeResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    knowledge_id: int
    category: str
    title: str
    response_code: int = Field(description="响应代码")
    response_msg: str = Field(description="响应信息")
    process_status: str = Field(description="处理状态")
    processing_time: float = Field(description="处理耗时（秒）")


class DocumentResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    document_id: int
    category: str
    title: str
    knowledge_id: int
    file_type: str
    parse_status: str = Field(description="文档解析状态：pending / processing / completed / failed")
    response_code: int = Field(description="响应代码")
    response_msg: str = Field(description="响应信息")
    process_status: str = Field(description="本次请求处理状态")
    processing_time: float = Field(description="处理耗时（秒）")


class KnowledgeItem(BaseModel):
    knowledge_id: int
    title: str
    category: str


class KnowledgeListResponse(BaseModel):
    knowledge_bases: List[KnowledgeItem]
    total: int


class DocumentItem(BaseModel):
    document_id: int
    title: str
    category: str
    file_type: str
    parse_status: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total: int
    knowledge_id: int


class RAGRequest(BaseModel):
    knowledge_id: int
    message: List[Dict]


class RAGResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Dict]
    response_code: int = Field(description="响应代码")
    response_msg: str = Field(description="响应信息")
    process_status: str = Field(description="处理状态")
    processing_time: float = Field(description="处理耗时（秒）")
