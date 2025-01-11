import pandas as pd

# Path to your .txt file (change this to the actual file path)
file_path = "Toto658.txt"

# Read the data from the .txt file into a DataFrame
df = pd.read_csv(file_path)

# Renaming columns to match the structure you want
df.rename(columns={
    "DrawNo": "Draw_Number",
    "DrawnNo1": "Winning_Number_1",
    "DrawnNo2": "Winning_Number_2",
    "DrawnNo3": "Winning_Number_3",
    "DrawnNo4": "Winning_Number_4",
    "DrawnNo5": "Winning_Number_5",
    "DrawnNo6": "Winning_Number_6"
}, inplace=True)

# Optionally, you can remove the 'DrawDate' and 'Jackpot' columns if you don't need them
df.drop(columns=["DrawDate", "Jackpot"], inplace=True)

# Save to a new CSV file (optional)
df.to_csv("lottery_data.csv", index=False)

# Display the DataFrame
print(df.head())