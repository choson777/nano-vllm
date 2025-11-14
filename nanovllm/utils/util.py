import uuid
from nanovllm import Request

class StepOutput:
    def __init__(self, req: Request, new_token: str, new_token_id: int | list[int]):
        self.request_id = req.request_id
        self.request = req
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = req.is_finished
    
    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"request={self.request},"
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )




def random_uuid() -> str:
    return str(uuid.uuid4().hex)