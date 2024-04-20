"""Generates leaderboard."""

import re

import pandas as pd

# Read the markdown table into a DataFrame
with open("session_2/challenge/leaderboard.md", "r") as file:
    content = file.read()

start_marker = "<!-- leader-board-begins -->\n"
start_index = content.find(start_marker)
end_index = content.find("\n<!-- leader-board-ends -->")
table_content = content[start_index:end_index]


# Extract rows using regex
rows = re.findall(
    r"\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|", table_content
)[2:]

# Create a DataFrame from the extracted rows
df = pd.DataFrame(
    rows,
    columns=[
        "Rank",
        "Profile Image",
        "GitHub Username",
        "Solution",
        "Accuracy %",
    ],
)

# Strip extra spaces before and after text in each cell
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert "Rank" column to integer and "Accuracy %" column to float
df["Rank"] = df["Rank"].astype(int)
df["Accuracy %"] = df["Accuracy %"].astype(float)

# Add a new entry to the DataFrame
new_entry = {
    "Rank": len(df) + 1,
    "Profile Image": '<img src="https://github.com/hetul-patel.png" width="50px" height="50px" class="profile-image">',
    "GitHub Username": "[New User](https://github.com/new_user)",
    "Solution": "[New Solution](https://github.com/new_solution)",
    "Accuracy %": 99.5,
}  # Example accuracy value

df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

# Keep only the highest submission for each user
highest_indices = df.groupby("GitHub Username")["Accuracy %"].idxmax()
df_highest = df.loc[highest_indices]

# Sort the DataFrame by "Accuracy %" column in descending order
df_sorted = df_highest.sort_values(
    by="Accuracy %", ascending=False
).reset_index(drop=True)

# Update the "Rank" column after sorting
df_sorted["Rank"] = df_sorted.index + 1

# Convert the DataFrame back to markdown format
markdown_table = df_sorted.to_markdown(index=False)

# Replace the existing table in the markdown file with the sorted table
new_content = (
    content[: start_index + len(start_marker)]
    + markdown_table
    + content[end_index:]
)

# Write the updated content back to the markdown file
with open("session_2/challenge/leaderboard.md", "w") as file:
    file.write(new_content)
