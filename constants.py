INTENT_KEYWORDS = {
    'mean_balance': [
        'average balance',
        'mean balance',
        'avg balance'
    ],
    'count_default': [
        'how many default',
        'number of defaults',
        'clients defaulted'
    ],
    'filter_clients': [
        'filter clients',
        'show clients',
        'only clients'
    ],
    'count_subscribed': [
        'how many subscribed',
        'clients said yes',
        'who said yes',
        'subscribed to term deposit',
        'subscribed'
    ],
    'filter_job_marital': [
        'unemployed',
        'married',
        'single',
        'job',
        'marital'
    ]
}

QUERIES = [
    "What's the average balance of married clients?",
    "How many clients defaulted on their loans?",
    "Filter clients who are unemployed and single.",
    "How many clients subscribed to the term deposit?",
    "Show me only clients with job management."
]

COLUMN_MATCH_THRESHOLD = 70

INTENT_HANDLERS = {
    'mean_balance': 'handle_mean_balance',
    'count_default': 'handle_count_default',
    'filter_clients': 'handle_filter_clients',
    'count_subscribed': 'handle_count_subscribed',
    'filter_job_marital': 'handle_filter_job_marital'
}