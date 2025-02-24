import pandas as pd

data = {
    'Name': ['abc', 'efg', ' hgt', 'hju', 'rug', 'uht', 'ujn', 'oka', 'mat', 'lop'],
    'Age': ['20','22','30','18','36', '40', '15', '19', '18', '20'],
    'PhoneNo': ['9876543210', '9638527410', '7539518520',' 9521753852', '9874106352', '9637410852', '9070804050', '9876543210',' 9876543210', '9630147852'],
    'Address': ['Chennai', 'Delhi', 'Banglore', 'Kolkata', 'Mumbai', 'Bhopal', 'Hyderabad', 'Chittoor', 'Ahmadabd', 'Delhi'],
    'BloodGroup': ['O+', 'O-', 'AB+', 'A+', 'AB-', 'O-', 'A-', 'B-', 'B+', 'O+']
}

data2 = {
    'Name': ['abc', 'efg', ' hgt', 'hju', 'rug', 'uht', 'ujn', 'oka', 'mat', 'lop'],
    'Occupation': ['Doctor', 'Developer', 'Engineer', 'Driver', 'Cook', 'Businessmen', 'Teacher', 'Student', 'Doctor','Traveller'],
    'Email': ['abc@gmail.com', 'efg@gmail.com', ' hgt@gmail.com', 'hju@gmail.com', 'rug@gmail.com', 'uht@gmail.com', 'ujn@gmail.com', 'oka@gmail.com', 'mat@gmail.com', 'lop@gmail.com'],
}

data_Layout = pd.DataFrame(data)
data_Layout2 = pd.DataFrame(data2)
# print(data_Layout)

# print(data_Layout.head())

# print(data_Layout.tail())

merge = pd.merge(data_Layout, data_Layout2, on='Name')
print(merge)