from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    age:Optional[int]=None # default value
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10)

new_student =  {'name':'Sathwik','age':21,'email':'abccdd@gmail.com','cgpa':8}
student = Student(**new_student)

print(student)