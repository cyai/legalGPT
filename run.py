from legalGPT.utils import qa_pipeline

chian = qa_pipeline()

def main():
    user_input = input("Enter your question: ")

    output = chian(user_input)

    print(f"Result: {output['result']}")

    print(f"Source documents: {output['source_documents']}")
    
if __name__ == "__main__":
    main()