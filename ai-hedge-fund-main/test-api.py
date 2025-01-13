import requests

# add your API key to the headers
headers = {
    "X-API-KEY": "6419c5be-36cc-4e62-b60b-288d9f0bc198"
}

# set your query params
ticker = 'NVDA'     # stock ticker
period = 'annual'   # possible values are 'annual', 'quarterly', or 'ttm'
limit = 30          # number of statements to return

# create the URL
url = (
    f'https://api.financialdatasets.ai/financials/'
    f'?ticker={ticker}'
    f'&period={period}'
    f'&limit={limit}'
)

# make API request
response = requests.get(url, headers=headers)

# parse financials from the response
financials = response.json().get('financials')
print(financials)
print('-'*25)

# get income statements
income_statements = financials.get('income_statements')
print(income_statements)
print('-'*25)

# get balance sheets
balance_sheets = financials.get('balance_sheets')
print(balance_sheets)
print('-'*25)

# get cash flow statements
cash_flow_statements = financials.get('cash_flow_statements')
print(cash_flow_statements)
print('-'*25)
