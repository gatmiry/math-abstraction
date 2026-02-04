"""
Minimal code to generate solutions using OpenAI API for geometry problems from math dataset.
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
import datasets
from tqdm import tqdm


class GeometryProblemGenerator:
    """Generate solutions to geometry problems using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> datasets.Dataset:
        """Load the math dataset. Defaults to competition_math from HuggingFace."""
        if dataset_path is None:
            dataset_path = "qwedsacf/competition_math"
        
        #if os.path.exists(dataset_path):
        #    return datasets.load_from_disk(dataset_path)
        return datasets.load_dataset(dataset_path, split="train")
    
    def filter_problems(
        self,
        dataset: datasets.Dataset,
        max_problems: Optional[int] = None,
        problem_type: str = "Geometry",
        level: str = "Level 5"
    ) -> List[Dict]:
        """Filter dataset for geometry and easy problems using dataset fields."""
        filtered = []
        
        for example in tqdm(dataset, desc="Filtering"):
            # competition_math dataset uses "type" field (e.g., "Geometry")
            if problem_type and example.get("type", "") != problem_type:
                continue
            
            # competition_math dataset uses "level" field (e.g., "Level 1" to "Level 5")
            if level and example.get("level", "") != level:
                continue
            
            # competition_math dataset uses "solution" field, not "answer"
            if not example.get("problem") or not example.get("solution"):
                continue
            
            filtered.append({
                "problem": example["problem"],
                "answer": example["solution"]  # Map solution to answer for compatibility
            })
            
            if max_problems and len(filtered) >= max_problems:
                break
        
        return filtered
    
    def generate_solution(self, problem: str, max_tokens: int = 2000) -> str:
        """Generate solution using OpenAI API with structured proof format."""
        proof_developer_prompt = """Generate a detailed, step-by-step mathematical proof for the given math problem. For each step, output the reasoning using exactly one of two tags:  
- **<LEMMA_THEOREM_TAG>…</LEMMA_THEOREM_TAG>:** Use this tag for any step that cites a lemma or theorem from Wikipedia. Inside this tag, include:  
   - the statement of the lemma/theorem,  
   - the name of the lemma/theorem,  
   - the relevant Wikipedia page address (URL),  
   - the mathematical topic (e.g., geometry, algebra, topology, probability, combinatorics, optimization, etc.).
- **<INTERMEDIATE_DERIVATION_TAG>…</INTERMEDIATE_DERIVATION_TAG>:** Use this tag for reasoning or logical derivation steps that link information together, interpret results, or manipulate equations based on previous steps or lemmas/theorems.

Continue the step-by-step proof until the answer is achieved. Write the final answer in the \boxed{} math environment, as is standard in mathematical problem-solving and competitions.

**Guidelines:**
- Ensure that each reasoning process and justification appears **before** any conclusions or derivations.  
- Each tag block should encompass one logical or justification step.
- Alternate using <LEMMA_THEOREM_TAG> and <INTERMEDIATE_DERIVATION_TAG> as appropriate to properly structure the proof.

**Output Format:**  
- Output the proof as a sequence of step blocks, each surrounded by the appropriate tags.  
  - The content within <LEMMA_THEOREM_TAG> should also include four fields: "statement", "name", "wikipedia_url", "topic".
  - The final answer should appear in a separate line at the end, within \boxed{}.
- Do not provide explanations outside the tags.
- Use plain text with tags (no code blocks).
- Length: As detailed as needed to justify each inference and transition, breaking the reasoning into justifiable steps per above.

**EXAMPLE:**

_Input math problem:_  
Prove that the sum of the interior angles of a triangle is 180 degrees.

_Output:_  
<LEMMA_THEOREM_TAG>  
statement: "The sum of the angles in an n-sided polygon is given by (n-2)×180 degrees."  
name: "Sum of Interior Angles of a Polygon"  
wikipedia_url: "https://en.wikipedia.org/wiki/Polygon#Angles"  
topic: "geometry"  
</LEMMA_THEOREM_TAG>  
<INTERMEDIATE_DERIVATION_TAG>  
Since a triangle has n=3 sides, applying the formula: (3-2)×180 = 1×180 = 180 degrees.  
</INTERMEDIATE_DERIVATION_TAG>  
\boxed{180^\circ}

**Note:** In a real answer, proofs for more complex problems would include several <LEMMA_THEOREM_TAG> and <INTERMEDIATE_DERIVATION_TAG> blocks, with more detailed content in each.

---
**Important Instructions & Objective Reminder:**  
For every input math problem, construct your answer as a sequence of step-by-step tagged blocks:
- Use <LEMMA_THEOREM_TAG> with statement, name, Wikipedia URL, and topic for definitions/theorems/lemmas from Wikipedia.
- Use <INTERMEDIATE_DERIVATION_TAG> for logical or deductive reasoning, connecting results.
- Always write the final answer in the \boxed{} environment as a standalone line.  
Expand proofs into as many blocks as required for clarity and correctness."""
        
        try:
            response = self.client.responses.create(
            prompt={
                 "id": "pmpt_69475c3f5c9c8194b0a136dc663d15b30cc8e7bedee1d50b",
                 "version": "2"
            },
            input=[{"role": "user", "content": f"{problem}"}],
            reasoning={
                "summary": "auto"
            },
            include=[
                "reasoning.encrypted_content",
                "web_search_call.action.sources"
            ]
            )
            return response.output_text #response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_solutions_batch(
        self,
        problems: List[Dict],
        output_path: Optional[str] = None,
        dataset_output_path: Optional[str] = None
    ) -> List[Dict]:
        """Generate solutions for a batch of problems."""
        results = []
        
        for i, p in enumerate(tqdm(problems, desc="Generating")):
            solution = self.generate_solution(p["problem"])
            result = {
                "problem": p["problem"],
                "ground_truth": p["answer"],
                "solution": solution
            }
            results.append(result)
        
        # Save as JSONL
        if output_path:
            self._save_jsonl(results, output_path)
        
        # Save as HuggingFace Dataset
        if dataset_output_path:
            self._save_dataset(results, dataset_output_path)
        
        return results
    
    def _save_jsonl(self, results: List[Dict], path: str):
        """Save results to JSONL."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    
    def _save_dataset(self, results: List[Dict], path: str):
        """Save results as HuggingFace Dataset."""
        dataset = datasets.Dataset.from_list(results)
        dataset.save_to_disk(path)
        print(f"Saved dataset to {path}")


def main():
    generator = GeometryProblemGenerator(model="gpt-4o")
    
    # Load competition_math dataset (default)
    dataset = generator.load_dataset()
    problems = generator.filter_problems(
        dataset,
        max_problems=10,
        problem_type="Geometry",
        level="Level 5"  # Easiest level
    )
    
    results = generator.generate_solutions_batch(
        problems,
        output_path="outputs/math_solutions.jsonl",
        dataset_output_path="outputs/math_solutions_dataset"
    )
    
    print(f"Generated {len(results)} solutions")


if __name__ == "__main__":
    main()
