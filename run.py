from legalGPT.utils import qa_pipeline

chian = qa_pipeline()

def main():
    user_input = input("Enter your question: ")

    output = chian(user_input)

    print(output)