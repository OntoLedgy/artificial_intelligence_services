from agents.coding.coding_agent import AdvancedCodeAgent


class TestCodingAgent():


    def test_advanced_coding_agent(self):
        API_KEY = "Use Your Own Key Here"
    
        agent = AdvancedCodeAgent()
    
        print("🚀 Advanced Claude Code Agent with Validation")
        print("=" * 60)
    
        print("\n🔢 Example 1: Prime Number Analysis with Twin Prime Detection")
        print("-" * 60)
        query1 = """
        Find all prime numbers between 1 and 200, then:
        1. Calculate their sum
        2. Find all twin prime pairs (primes that differ by 2)
        3. Calculate the average gap between consecutive primes
        4. Identify the largest prime gap in this range
        After computation, validate that we found the correct number of primes and that all identified numbers are actually prime.
        """
        result1 = agent.run(query1)
        print(result1)