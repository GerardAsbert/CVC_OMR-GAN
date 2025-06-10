import statistics

# Original list
old_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Example list

# Get means of every 10 items
new_list = [statistics.mean(old_list[i:i+10]) for i in range(0, len(old_list), 10)]

print(new_list)