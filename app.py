import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Optional



class TextSummarizationAgent:
    def __init__(self, api_key: Optional[str] = None):
      
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("Google API Key must be set either via parameter or GOOGLE_API_KEY environment variable")
        

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            top_p=0.85
        )
        
        #Here prompt templates...........................
        self.summarization_prompt = PromptTemplate(
            input_variables=["text", "style", "max_length"],
            template="""You are an expert summarizer. Your task is to create a {style} summary of the following text, 
            ensuring the summary does not exceed {max_length} words while preserving the key information:

            Text: ```{text}```

            Summary requirements:
            1. Capture the main ideas and key points
            2. Maintain the original context and tone
            3. Be concise and clear
            4. Do not include any external information not present in the original text

            Summary:"""
        )
    
    def summarize(
        self, 
        text: str, 
        style: str = "neutral", 
        max_length: int = 100
    ) -> str:
        """
        Generate a summary of the given text
        
        Args:
            text (str): Input text to summarize
            style (str): Summary style (neutral, concise, detailed)
            max_length (int): Maximum number of words in summary
        
        Returns:
            str: Generated summary
        """
        # Validate inputs
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Create summarization chain
        summarization_chain = (
            self.summarization_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Generate summary
        summary = summarization_chain.invoke({
            "text": text, 
            "style": style, 
            "max_length": max_length
        })
        
        return summary
    
    def batch_summarize(
        self, 
        texts: List[str], 
        style: str = "neutral", 
        max_length: int = 100
    ) -> List[str]:
        """
        Summarize multiple texts in batch
        
        Args:
            texts (List[str]): List of texts to summarize
            style (str): Summary style
            max_length (int): Maximum words per summary
        
        Returns:
            List[str]: List of generated summaries
        """
        return [self.summarize(text, style, max_length) for text in texts]

def main():
    # Example usage
    api_key = "AIzaSyAl9F8on5RnIiEOs9wrShQRlhcHn9gIeYc" 

    # Sample texts for summarization
    texts = [
        """
        Artificial Intelligence is transforming multiple industries by providing 
        intelligent solutions to complex problems. Machine learning algorithms 
        can now predict outcomes, automate processes, and generate insights 
        that were previously impossible. From healthcare to finance, AI is 
        driving innovation and efficiency across various sectors.
        """,
        """
        Climate change represents one of the most significant challenges 
        facing our planet. Rising global temperatures, increasing extreme 
        weather events, and melting polar ice caps demonstrate the urgent 
        need for comprehensive environmental strategies and sustainable 
        development practices.
        """
    ]
    
    # Initialize the summarization agent
    summarizer = TextSummarizationAgent(api_key)
    
    # Summarize single text
    print("Single Text Summary:")
    single_summary = summarizer.summarize(
        texts[0], 
        style="concise", 
        max_length=50
    )
    print(single_summary)
    
    # Batch summarization
    print("\nBatch Summaries:")
    batch_summaries = summarizer.batch_summarize(
        texts, 
        style="neutral", 
        max_length=75
    )
    for summary in batch_summaries:
        print(summary)

if __name__ == "__main__":
    main()