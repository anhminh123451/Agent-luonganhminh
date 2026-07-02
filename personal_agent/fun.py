from tenacity import stop
arr = [1,2,3,4,5,6,7,8,9]
new_arr = []

n = len(arr)
for i in range(n-1,-1,-1):
    new_arr.append(arr[i])

print(new_arr)
    