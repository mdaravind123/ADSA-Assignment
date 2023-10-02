#insertion sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

input_array = [5,2,9,1,6]
insertion_sort(input_array)
print("Sorted array:", input_array)

#Time complexity Analysis
#1. Best-case time complexity: O(N)
#2. Average-case time complexity: O(N^2)
#3. Worst-case time complexity: O(N^2)
#Insertion Sort performs well on small datasets

#Comparison with Quick Sort:
#Quick Sort is generally faster than Insertion Sort for larger datasets and has an average-case time complexity of O(N log N).

