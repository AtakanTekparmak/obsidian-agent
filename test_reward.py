from training.reward.reward import get_reward
from data.schemas.kb import Fact

FACTS_TO_CHECK = [
    Fact(
        fact_description_or_change="Name: Sofie Jansen",
        timestamp=None
    ),
    Fact(
        fact_description_or_change="Age: 22",
        timestamp=None
    ),
    Fact(
        fact_description_or_change="Gender: female",
        timestamp=None
    ),
    Fact(
        fact_description_or_change="Birthplace: Utrecht, Netherlands",
        timestamp=None
    ),
    Fact(
        fact_description_or_change="Grew up with a keen interest in technology's potential to solve societal problems.",
        timestamp=None
    )
]

REPO_DUMP = """
DIRECTORY STRUCTURE:
character_profile/
├── background_story.md
└── personal_details.md

FILE CONTENTS:

════════ character_profile/background_story.md ════════
# Sofie Jansen's Background

Sofie Jansen grew up in the Netherlands with a keen interest in technology's potential to solve societal problems. This early fascination shaped her academic and career aspirations. She believes strongly in leveraging innovation for social good.


--------------------------------------------------------------------------------


════════ character_profile/personal_details.md ════════
# Personal Details: Sofie Jansen

Here is some basic information about Sofie:

- **Name:** Sofie Jansen
- **Gender:** female
- **Birthplace:** The Netherlands

## Additional Notes
Sofie is a dynamic individual with a clear vision for her future.


--------------------------------------------------------------------------------
"""

reward = get_reward(REPO_DUMP, FACTS_TO_CHECK)
print(reward)