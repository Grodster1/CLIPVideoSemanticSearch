from src.search_engine import SearchEngine
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_ROOT, 'data', 'index')

def main():
    engine = SearchEngine(INDEX_DIR)
    print("\n----Video Semantic Search----")
    print("Type 'exit' to quit")
    
    
    while True:
        query = input("Please enter search query in English: ")
        if query.lower() == 'exit':
            break
        
        results = engine.search(query, top_n=5)
        
        print(f"RESULTS FOR '{query}' query:")
        for i, result in enumerate(results):
            timestamp = result['timestamp']
            score = result['score']
            minutes = timestamp//60
            seconds = timestamp%60
            time = f"{int(minutes):02d}:{int(seconds):02d}"
            
            print(f"{i+1}. Score: {score:.4f} (cos similarity) Time: {time}")
        
        print('-'*20)
        
if __name__ == '__main__':
    main()    