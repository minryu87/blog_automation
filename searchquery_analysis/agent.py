import os
import json
import re
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv(find_dotenv())

LLM = Gemini(
    id=os.getenv("GEMINI_LITE_MODEL", "gemini-2.5-pro"),
    api_key=os.getenv("GEMINI_API_KEY")
)

class TaxonomyRefinementAgent:
    def __init__(self):
        self.agent = Agent(
            name="TaxonomyRefinementAgent",
            role="An expert linguist and data analyst who refines search query classification taxonomies.",
            model=LLM,
            instructions=self._get_instructions(),
        )

    def _get_instructions(self):
        return [
            "Your task is to analyze a given unclassified search term ('expression') and decide how to integrate it into an existing taxonomy for dental clinic search queries.",
            "You have three possible actions: MAP_TO_EXISTING, CREATE_NEW, or IGNORE.",
            "Carefully review the existing `taxonomy_schema` before making a decision.",
            
            "**Decision Logic:**",
            "1. **MAP_TO_EXISTING**: Choose this if the expression is a clear synonym, hyponym, or a closely related term to an *existing* label. Provide the `target_label` in 'Category:Label' format.",
            "   - Example Expression: '깨짐' -> This is a synonym for '파손'. Your action should be to map it to '증상:파손/깨짐'.",
            "   - Example Expression: '치아뿌리' -> This is a specific type of '잇몸'. Map it to '위치:잇몸'.",
            
            "2. **CREATE_NEW**: Choose this *only if* the expression represents a significant, distinct concept not covered by the existing taxonomy. Provide a `new_label` in 'NewCategory:NewLabel' format.",
            "   - Example Expression: '미세현미경' -> This is a specific piece of equipment. The taxonomy lacks a category for tools. Your action should be to create a new label like '진단장비:미세현미경'.",
            "   - Example Expression: '치과' -> This is a crucial but unclassified term representing a place. Create a new label '장소:의료기관'.",

            "3. **IGNORE**: Choose this *only for* expressions that are clearly grammatical particles, stopwords, typos, or too generic to be useful for marketing analysis (e.g., '의', '가', '것', '한').",

            "**Output Format:**",
            "You MUST output a single, valid JSON object and nothing else. Do not include markdown formatting like ```json.",
            "The JSON object must have the following structure:",
            "{",
            "  \"thought\": \"Your brief reasoning for the decision.\",",
            "  \"action\": \"MAP_TO_EXISTING | CREATE_NEW | IGNORE\",",
            "  \"target_label\": \"(Required if action is MAP_TO_EXISTING) e.g., '증상:파손/깨짐'\",",
            "  \"new_label\": \"(Required if action is CREATE_NEW) e.g., '장소:의료기관'\"",
            "}"
        ]

    def _extract_json(self, text):
        # LLM 응답에서 JSON 블록을 안정적으로 추출
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def decide(self, expression, taxonomy):
        # Format the taxonomy for the prompt
        taxonomy_schema = json.dumps(taxonomy, indent=2, ensure_ascii=False)
        
        prompt = f"""
        **Expression to Analyze:**
        `{expression}`

        **Existing Taxonomy Schema:**
        ```json
        {taxonomy_schema}
        ```
        
        Based on the expression and the existing schema, what is your decision?
        """
        
        try:
            response_str = self.agent.run(prompt).content
            response_json = self._extract_json(response_str)
            
            if response_json:
                return response_json
            else:
                print(f"Error: Could not parse JSON from agent response for '{expression}'. Response: {response_str}")
                return {"thought": "Failed to parse JSON from LLM response.", "action": "IGNORE"}

        except Exception as e:
            print(f"Error calling LLM agent for expression '{expression}': {e}")
            # Fallback to IGNORE on error to prevent system crash
            return {
                "thought": f"Agent failed with error: {e}",
                "action": "IGNORE"
            } 