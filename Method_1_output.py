import pandas as pd
from math import log10

file_path = "outlistFreq3000.csv"

df = pd.read_csv(file_path)
df = df.drop_duplicates(subset=["0"])
df = df.set_index("0")["2"]


def wordFinder(word, df):
    """
    Find a word in the dataset df and return its value from column 3.
    If the word doesn't exist, return 0.

    Parameters:
    word (str): The word to search for.
    df (pandas.DataFrame): The dataset to search in.

    Returns:
    int or float: The value from column 3 if the word is found, 0 otherwise.
    """
    # Check if the word exists in the dataset
    if word in df.index:
        return df.loc[word]
    else:
        # Word doesn't exist, return 0
        return 0


def sumUpp(input_string, df):
    """
    Extract each word in the input string and apply the wordFinder function to each word.
    Sum up the returned values from wordFinder for each word.

    Parameters:
    input_string (str): The input string containing words.
    df (pandas.DataFrame): The dataset to search in.

    Returns:
    int or float: The sum of the returned values from wordFinder for each word.
    """
    
    # Split the input string into words
    words = input_string.split()
    
    # Initialize sum
    total_sum = 0
    num_words = 0
    
    # Iterate over each word
    for word in words:
        # Apply wordFinder function to each word and add the returned value to the total sum
        total_sum += wordFinder(word, df)
        num_words = num_words +1
    
    
    total_sum = log10(total_sum)/num_words
    if total_sum <= 0: 
        total_sum = 0
    
    total_sum = round(total_sum, 3)
    return total_sum



print("Are you as funny as you think you are? ")
input_string = input("Enter a joke: ")

result = sumUpp(input_string,df)

# Output the result
print("Here's how funny you are:", result)