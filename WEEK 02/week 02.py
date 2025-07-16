def remove_duplicates(data_list):
    unique_data = []
    seen = set()

    for record in data_list:
        # Convert record to tuple for comparison
        identifier = (record['name'], record['age'])
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(record)
    
    return unique_data

def filter_valid_data(data_list):
    # Only include records with valid (non-negative) ages
    return [record for record in data_list if record['age'] >= 0]

def calculate_average_age(data_list):
    total_age = sum(record['age'] for record in data_list)
    return total_age / len(data_list) if data_list else 0

# Sample data
data = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": -1},
    {"name": "Alice", "age": 25},
    {"name": "Charlie", "age": 30},
    {"name": "Daisy", "age": 22},
    {"name": "Bob", "age": -1}
]

# Cleaning process
data_no_duplicates = remove_duplicates(data)
valid_data = filter_valid_data(data_no_duplicates)
average_age = calculate_average_age(valid_data)

# Output results
print("Cleaned Data:")
for record in valid_data:
    print(record)

print("\nAverage Age of Valid Users:", average_age)
