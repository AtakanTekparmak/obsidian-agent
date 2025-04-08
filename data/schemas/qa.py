from typing import List
from data.schemas.base import BaseSchema
from data.schemas.kb import KnowledgeBase

class QuestionAnswer(BaseSchema):
    question: str
    correct_answer: str

class QuestionAnswers(BaseSchema):
    questions: List[QuestionAnswer] 

class PersonaQuestionAnswers(BaseSchema):
    name_surname: str
    questions: List[QuestionAnswer] 

class QuestionAnswersList(BaseSchema):
    question_answers: List[PersonaQuestionAnswers]