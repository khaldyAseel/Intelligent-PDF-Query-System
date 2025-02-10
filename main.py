from together import Together
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key="339d6aae5dd24ebfff4eb075e953efa0c8708bf7aeae13ba455b06f1b879753b")

query = "What factors contribute to price volatility in the cocoa market?"

context = """Cocoa beans have to get from the many small farmers, who are often in remote
areas of developing countries, to the cocoa processing factories that may be
located in temperate countries. They can pass through a number of intermediar-
ies, each of whom plays an important role. This section describes the steps in the
chain, the impact on quality and how the price is determined. The next section


looks at the cocoa value chain and the issue of farmer poverty. The price of cocoa is
given in US$ or GB£ per tonne and is determined in the open markets of New York
and London. The evolution of prices, production and consumption (demand or
“grindings”) is given in Figure 2.14. From this graph one can note that production
and consumption are closely balanced and have grown steadily at the same rate.
However, prices are more volatile and are influenced by production, consumption,
stock levels, political, social and economic factors and speculator activity.
"""

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
      {"role": "system", "content": "You are a helpful chatbot."},
      {"role": "user", "content": f"Answer the question: {query}. Use only information provided here: {context}"},
    ],
)

print(response.choices[0].message.content)
