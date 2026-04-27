# Get icelandic
import os
import json
import pandas as pd
import csv

# From EPhishGen
import re
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
import warnings

def get_ice(filename="data/icelandic_llm_emails.json"):    
    all_emails = set()
    
    with open(filename, 'r', encoding='utf-8') as f:
        emails = json.load(f)
        
        print(f"Loaded data succesfully ({len(emails)} emails) from {filename}")
        
        for email_dict in emails:
          # Convert the dictionary to a frozenset of its items to make it hashable
          hashable_email = frozenset(email_dict.items())
          all_emails.add(hashable_email)
    
    all_emails_dicts = []
    for email_frozenset in all_emails:
        all_emails_dicts.append(dict(email_frozenset))
    
    # Create a Pandas DataFrame from the list of dictionaries
    emails_df = pd.DataFrame(all_emails_dicts)
    
    stats(emails_df)

    return emails_df

def get_ephish(filename="data/ephish_emails.json"):    
    all_emails = set()
    
    with open(filename, 'r', encoding='utf-8') as f:
        emails = json.load(f)
        
        print(f"Loaded data succesfully ({len(emails)} emails) from {filename}")
        
        for email_dict in emails:
          # Convert the dictionary to a frozenset of its items to make it hashable
          hashable_email = frozenset(email_dict.items())
          all_emails.add(hashable_email)
    
    all_emails_dicts = []
    for email_frozenset in all_emails:
        all_emails_dicts.append(dict(email_frozenset))
    
    # Create a Pandas DataFrame from the list of dictionaries
    emails_df = pd.DataFrame(all_emails_dicts)
    emails_df = emails_df[emails_df['Language'] == 'en'].reset_index(drop=True)
    emails_df= emails_df.drop(columns=['Language'])
    
    stats(emails_df)

    return emails_df

# From EPhishGen
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def bert_cleaning(text):
    if pd.isna(text):
        return ""
    # Remove HTML, Non-ASCII, and extra whitespace
    text = BeautifulSoup(text, "html.parser").get_text()  
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)          
    text = re.sub(r'\s+', ' ', text).strip()             
    return text

def get_known(folder="data/ephishgen_datasets/datasets/datasets_BERT"):
    all_dataframes = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)

                    # 1. Standardize column names to: subject, body, type
                    # Mapping: label/type -> type | text/tokenized_body -> body
                    rename_map = {
                        'label': 'type',
                        'text': 'body',
                        'tokenized_body': 'body'
                    }
                    df = df.rename(columns=rename_map)

                    # 2. Ensure essential columns exist
                    if 'body' not in df.columns or 'type' not in df.columns:
                        continue
                    if 'subject' not in df.columns:
                        df['subject'] = "" # Placeholder if missing

                    # 3. Clean labels (type)
                    df["type"] = pd.to_numeric(df["type"], errors="coerce")
                    df = df.dropna(subset=["type"]).copy()
                    df["type"] = df["type"].astype(int)

                    # 4. Filter for valid content
                    df = df[
                        df["body"].apply(lambda x: isinstance(x, str) and x.strip() != "")
                    ].copy()

                    # 5. Apply your cleaning logic
                    df["Subject"] = df["subject"].apply(bert_cleaning)
                    df["Body"] = df["body"].apply(bert_cleaning)

                    # 6. De-duplicate and validate class balance
                    df = df.drop_duplicates(subset=["Subject", "Body"]).reset_index(drop=True)
                    
                    if df["type"].nunique() >= 2:
                        # Keep only the requested columns
                        all_dataframes.append(df[["Subject", "Body", "type"]])
                        print(f"Loaded {file}: {len(df)} samples")
                    else:
                        print(f"Skipped {file}: Missing class variety")

                except Exception as e:
                    print(f"Error in {file}: {e}")

    if not all_dataframes:
        df_final = pd.DataFrame(columns=["Subject", "Body", "type"])
        stats(df_final)

        return df_final


    df_final = pd.concat(all_dataframes, ignore_index=True)
    stats(df_final)

    return df_final

def stats(df):
    type_counts = df['type'].value_counts()

    print("Count of emails by 'type':")
    print(type_counts)
    
    # Calculate percentages
    total_emails = type_counts.sum()
    percentage_type_0 = (type_counts.get(0, 0) / total_emails) * 100
    percentage_type_1 = (type_counts.get(1, 0) / total_emails) * 100
    
    print("\nPercentage breakdown:")
    print(f"Normal emails (type 0): {percentage_type_0:.2f}%")
    print(f"Phishing emails (type 1): {percentage_type_1:.2f}%")
    print()