import pytest
from tokenisation.tokeniser import count_tokens_in_string
import seaborn as sns
import matplotlib.pyplot as plt


class TestChunking:
    @pytest.fixture(autouse=True)
    def setup(self,
              news_data):
        self.news = news_data
    
    
    def test_chunking_news_articles(
            self):
        self.news["tokens"] = [
            count_tokens_in_string(
                    f"{row['title']}"
                    f"{row['text']}")
            
            for i, row in self.news.iterrows()
            ]
        
        self.news.head()
        
        sns.histplot(
                self.news["tokens"],
                kde=False)
        plt.title(
            "Distribution of chunk sizes")
        plt.xlabel(
            "Token count")
        plt.ylabel(
            "Frequency")
        plt.show()