# app/prompts/memory_prompts.py
"""
기억 생성 프롬프트 - 함수 방식
"""

from typing import Dict

def get_keepsake_memory_prompt(keepsake_data: Dict, deceased_info: Dict) -> str:
    return f"""
이 유품은 고인과의 추억이 담긴 물건이야.
고인의 배경 설명보다는 **이 유품을 통해 떠오르는 구체적인 기억이나 감정**에 집중해줘.

 유품 정보:
- 이름: {keepsake_data.get('item_name', '알 수 없음')}
- 설명: {keepsake_data.get('description', '없음')}
- 특별한 이야기: {keepsake_data.get('special_story', '없음')}
- 획득 시기: {keepsake_data.get('acquisition_period', '모름')}
- 이미지 경로: {keepsake_data.get('image_path', '')}

 작성 방식:
- 유품을 바라보며 떠오르는 장면, 대화, 감정 등을 진심 어린 어조로 3~4문장으로 작성해줘.
- 유품이 어떤 순간을 상기시키는지, 내가 왜 그것을 간직하고 싶은지 중심으로 말해줘.
- 고인의 성격, 취미, 나이 등은 최소한으로만 반영하고, **유품 자체의 기억**에 집중해줘.

예시:
"이 작은 필통은 엄마가 중학생 때 쓰시던 거래. 낡았지만 안쪽엔 엄마 글씨로 이름이 적혀 있어. 책상 위에 두면 항상 옆에 있는 느낌이 들어서 자주 들여다보곤 해."

 **단순 묘사가 아니라 ‘감정이 담긴 기억’ 문장만 작성해주세요. 설명, 구분선 등은 생략해주세요.**
"""

def get_photo_memory_prompt(photo_data: Dict, deceased_info: Dict) -> str:
    return f"""
이 사진은 고인과 함께한 실제 순간을 담고 있어.  
사진 정보만을 바탕으로 **기억나는 실제 경험이나 감정**을 떠올려 정리해줘.

 사진 정보:
- 제목: {photo_data.get('title', '제목 없음')}
- 촬영일: {photo_data.get('photo_date', '날짜 미상')}
- 설명: {photo_data.get('description', '설명 없음')}
- 이미지 경로: {photo_data.get('image_path', '')}

 작성 지침:
- 창작이 아니라, **실제 기억을 바탕으로** 작성해주세요.
- “사진을 보며 무슨 일이 있었는지 기억나는 장면, 분위기, 표정, 대화”를 3~4문장으로 묘사해주세요.
- ‘햇살이 들어왔다’, ‘향기가 어땠다’ 같은 과한 상상보다는, **사진 설명에 기반한 사실 중심 기억**을 정리해주세요.
- 감성은 담되, 실제 있었던 일처럼 느껴져야 합니다.

예시:
"이 사진은 엄마가 생일날 미역국을 끓여주시던 장면이야. 양옆에 있는 가족들이 다 함께 웃고 있었던 기억이 나. 엄마는 그날 하늘색 앞치마를 입고 있었고, 사진을 찍을 때 ‘웃어~’ 하고 외치셨지."

 중요한 점:
- 설명 형식이 아니라 “회상하는 문장”으로 작성해주세요.
- 설명, 태그, 날짜 등 메타데이터는 다시 언급하지 마세요.
"""