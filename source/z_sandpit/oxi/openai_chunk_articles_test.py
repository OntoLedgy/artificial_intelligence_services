import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from source.b_code.services.tokenisation.tokeniser import num_tokens_from_string

if __name__ == "__main__":
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )

    news["tokens"] = [
        num_tokens_from_string(f"{row['title']} {row['text']}")
        for i, row in news.iterrows()
    ]

    news.head()

    sns.histplot(news["tokens"], kde=False)
    plt.title("Distribution of chunk sizes")
    plt.xlabel("Token count")
    plt.ylabel("Frequency")
    plt.show()
