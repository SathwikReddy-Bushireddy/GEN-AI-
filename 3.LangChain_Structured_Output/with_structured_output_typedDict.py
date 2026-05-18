from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Literal, Optional, TypedDict,Annotated

load_dotenv()

hf_llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation",
)
model=ChatHuggingFace(llm=hf_llm)

# schema
class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down the key themes discussed in the review"]
    summary:Annotated[str,"A brief summary of the review"]
    satisfaction:Annotated[Literal["pos","neg"],"Return satisfaction of the review either negative,positive,neutral"]
    pros:Annotated[Optional[list[str]],"Write down all the pros inside the list "]
    cons:Annotated[Optional[list[str]],"Write down all the cons inside the list "]
    name:Annotated[Optional[str],"Return the name of the reviewer"]

structred_model=model.with_structured_output(Review)


res=structred_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Sathwik Reddy
""")

print(res)
# print(res['summary'])
# print(res['satisfaction'])