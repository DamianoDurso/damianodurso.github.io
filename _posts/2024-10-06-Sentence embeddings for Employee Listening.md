# The Modern Workhorse of Employee Listening: Leveraging Language Models for Better Surveys

**How often do you rely on surveys to measure employee engagement, customer satisfaction, or candidate fit?** In people analytics, these tools are indispensable. They help organizations gather insights and make informed decisions. However, building reliable surveys is complex and resource-intensive. Many fall back on poor measures due to the high cost, risk, and time involved.

This blog series will explore how **Large Language Model Embedding Psychometrics (LLMEP)** can streamline survey creation and make it easier, cheaper, and more effective. We’ll also delve into its coding so you can try it yourself!

In this first post, I’ll cover how **Large Language Models (LLMs)** can help transform survey questions into data for people analytics, enhancing accuracy and depth without requiring traditional data collection. Let’s dive in and discover how this applies to people analytics!

## Challenges in Traditional Survey Development

When it comes to creating surveys for people analytics, there are several hurdles that organizations often encounter:

- **Costly & Time-Consuming**: Crafting valid questions involves a considerable investment of resources. It requires in-depth research, rigorous testing, and a substantial dataset for validation. This process can be both financially burdensome and time-consuming.
- **High Failure Risk:** Even slight changes in the phrasing of questions can have a significant impact on the validity of responses. Additionally, administering multiple variations of surveys can lead to respondent fatigue and introduce bias into the results.

Given these challenges, the question arises: **Can we streamline this process without compromising the quality of the data gathered?** This is where LLMEP steps in. By addressing these challenges and providing a more efficient and effective approach to survey development, LLMEP offers a solution that not only simplifies the process but also allows to test the quality and validity of the questions before collecting any data.


## Enter Large Language Models (LLMs) and Their Power for People Analytics

**Large Language Models** such as GPT and BERT have transformed natural language processing. But how can they benefit people analytics? These models convert text into numerical representations known as *embeddings*, which capture sentence (or questions) meaning and can be further used to analyze survey data.

For people analytics, LLMs can:

- **Analyze Employee Feedback:** Quickly extract themes from open-ended responses.
- **Enhance Surveys with Semantics:** Create more accurate and insightful questions without extensive piloting.

In this article, we will focus on BERT and Sentence-BERT (SBERT), two LLMs that are particularly useful for transforming survey questions into analyzable data.

## Applying SBERT to People Analytics: A Practical Example

Imagine that you are developing a survey to measure workplace recognition, which you define as *"Workplace recognition is the acknowledgment and appreciation of an individual or a team's efforts, achievements, and contributions within the work environment. It can take the form of praise, rewards, or other forms of positive reinforcement to motivate and encourage employees."*

Let's say that, after some careful consideration, **you have managed to create a list of 10 questions** that you think are suitable to measure workplace recognition. However, this list is still too long, and different **stakeholders tell you that you can only use three questions because they don't want to overburden employees**. Also, you probably fear that by using too many questions the employees' motivaton to respond would drop, resulting in careless responding or other response biases, which would ultimately leave you with useless results.

Below I will show you in three simple steps **how you can use BERT and SBERT to pick the best three questions that would measure workplace recognition without collecting any data**.

## Step 1: Define the List of Questions and your construct definition


```python
# Define the construct for "Workplace Recognition"
construct = "Workplace recognition is the acknowledgment and appreciation of an individual or a team's efforts, achievements, and contributions within the work environment. It can take the form of praise, rewards, or other forms of positive reinforcement to motivate and encourage employees."

# List of questions about workplace recognition
questions = [
    "Do you feel acknowledged for your hard work by your manager?",
    "How often do you receive positive feedback for your contributions?",
    "Are your achievements celebrated within your team?",
    "Is your work recognized in team meetings?",
    "Do you receive public praise for your efforts?",
    "Are there rewards for outstanding performance in your workplace?",
    "Does your team leader show appreciation for a job well done?",
    "Are your contributions valued by your colleagues?",
    "Do you feel that your work is overlooked or undervalued?",
    "Is there a system in place for recognizing exceptional work in your company?"
]
```

## Step 2: Generate Embeddings for the Construct and Each Question


```python
# Import necessary libraries
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer

# Load BERT model and tokenizer for word embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize SBERT model for sentence embeddings
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to compute sentence embeddings using BERT (mean pooling)
def bert_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    word_embeddings = outputs.last_hidden_state.squeeze(0)[1:-1]
    return word_embeddings.mean(dim=0).numpy()

# Generate embeddings for the construct
construct_embedding_bert = bert_sentence_embedding(construct)
construct_embedding_sbert = sbert_model.encode(construct)

# Generate embeddings for each question
bert_embeddings = [bert_sentence_embedding(question) for question in questions]
sbert_embeddings = [sbert_model.encode(question) for question in questions]

```

## Step 3: Select the Best Three Questions Based on Cosine Similarity


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Calculate cosine similarities for BERT embeddings
similarities_bert = [cosine_similarity([construct_embedding_bert], [embedding])[0][0] for embedding in bert_embeddings]
# Calculate cosine similarities for SBERT embeddings
similarities_sbert = [cosine_similarity([construct_embedding_sbert], [embedding])[0][0] for embedding in sbert_embeddings]

# Find top 3 questions based on similarity scores from SBERT
top_indices_sbert = np.argsort(similarities_sbert)[-3:][::-1]
top_questions_sbert = [(questions[i], similarities_sbert[i]) for i in top_indices_sbert]

# Find top 3 questions based on similarity scores from BERT
top_indices_bert = np.argsort(similarities_bert)[-3:][::-1]
top_questions_bert = [(questions[i], similarities_bert[i]) for i in top_indices_bert]

# Display the top questions based on SBERT
print("Top 3 Questions (SBERT-based):")
for question, similarity in top_questions_sbert:
    print(f"Question: {question} | Similarity: {similarity:.4f}")

# Display the top questions based on BERT
print("\nTop 3 Questions (BERT-based):")
for question, similarity in top_questions_bert:
    print(f"Question: {question} | Similarity: {similarity:.4f}")

```

    Top 3 Questions (SBERT-based):
    Question: Are there rewards for outstanding performance in your workplace? | Similarity: 0.8215
    Question: Does your team leader show appreciation for a job well done? | Similarity: 0.8180
    Question: Do you feel acknowledged for your hard work by your manager? | Similarity: 0.7846
    
    Top 3 Questions (BERT-based):
    Question: Does your team leader show appreciation for a job well done? | Similarity: 0.7613
    Question: Is there a system in place for recognizing exceptional work in your company? | Similarity: 0.7508
    Question: Are there rewards for outstanding performance in your workplace? | Similarity: 0.7391


As we can see the only difference between SBERT and BERT is the second questions. **SBERT picks something that may resonate more with how people actually experience and value recognition in the workplace**. That is, rather than focusing on the existence of recognition structures, SBERT picks a question that is more likely to evoke feelings of appreciation and personal acknowledgment, **which seems to be supported by some [recent research](https://psycnet.apa.org/record/2008-05872-010)**. In any case, this is just a toy example, so you could probably find compelling arguments as well for having recognition systems in place.

## Conclusion: Unlocking Survey Insights with LLMEP in People Analytics

Using SBERT for survey item embedding provides a powerful tool for people analytics, allowing organizations to:

1. **Refine Survey Questions:** Ensure they measure the intended constructs without requiring extensive piloting.
2. **Analyze Open-Ended Feedback:** Transform question text into actionable data.
3. **Enhance Psychometric Accuracy:** Improve reliability and validity with less manual testing.

In the next blogpost I will show you how you can streamline this process even further by looking at multiple constucts, questions, and assessing the reliability and validity of your surveys data-free!

# References, appendix and additional info

- Guenole, N., Samo, A., Campion, J. K., Meade, A., Sun, T., & Oswald, F. (2024). Pseudo-Discrimination Parameters from Language Embeddings
- Guenole, N., D'Urso, E. D., Samo, A., & Sun, T. (2024). Pseudo Factor Analysis of Language Embedding Similarity Matrices: New Ways to Model Latent Constructs.
- Hommel, B. E., & Arslan, R. C. (2024). Language models accurately infer correlations between psychological items and scales from text alone.
- Wulff, D. U., & Mata, R. (2023). Automated jingle–jangle detection: Using embeddings to tackle taxonomic incommensurability.
