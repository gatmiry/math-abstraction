"""Simple example usage."""

from generate_solutions_openai import GeometryProblemGenerator


def main():
    generator = GeometryProblemGenerator(model="gpt-4o")
    
    # Load competition_math dataset (default) and filter
    dataset = generator.load_dataset()
    problems = generator.filter_problems(
        dataset,
        max_problems=10,
        problem_type="Geometry",
        level="Level 1"  # Easiest level
    )
    
    # Generate solutions
    results = generator.generate_solutions_batch(
        problems,
        output_path="outputs/solutions.jsonl",
        dataset_output_path="outputs/solutions_dataset"
    )
    
    print(f"Generated {len(results)} solutions")


if __name__ == "__main__":
    main()
