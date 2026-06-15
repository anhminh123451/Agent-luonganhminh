from pydantic import BaseModel, Field
import json

# 1. Định nghĩa cấu trúc bằng Pydantic Model
class Product(BaseModel):
    product_id: int = Field(description="Mã số sản phẩm")
    name: str = Field(min_length=3, description="Tên sản phẩm")
    in_stock: bool

# 2. Xuất ra JSON Schema chỉ với 1 dòng lện

dict = {
    "product_id": 123,
    "name": "Product 1",
    "in_stock": True,
    "price": 100
}
validate = Product(**dict)
# print(validate) 
schema = Product.model_json_schema()
print(schema)


