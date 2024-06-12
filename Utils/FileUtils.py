import csv
import torch

def write_mat(mat, out):
    with open(out, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in mat:
            vals = [float(val) for val in row]
            writer.writerow(vals)
        file.write('\n')

def write_1d(row, out):
    with open(out, mode='a', newline='') as file:
        writer = csv.writer(file)
        vals = [float(val) for val in row]
        writer.writerow(vals)   
        file.write('\n')     

def write_once(data, out):
    with open(out, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([float(val) for val in data])

def write_val(data, out):
    with open(out, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data])

def write_to_file(data, out):
    if len(data) == 0:
        return
    if data[0].dim() == 0:
        for d in data:
            write_val(d.item(), out)
    elif data[0].dim() == 1:
        for d in data:
            write_1d(d, out)
    else:
        for d in data:
            write_mat(d, out)

def clear_csv(out):
    with open(out, mode='w') as file:
        file.write('')

def read_matrices(input):
    matrices = []
    curr = []
    with open(input, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                curr.append([float(val) for val in row])
            else:
                if curr:
                    if len(curr) == 1:
                        matrices.append(torch.tensor(curr)[0])
                    else:
                        matrices.append(torch.tensor(curr))
                    curr = []
        
        if curr:
            matrices.append(torch.tensor(curr))
    return matrices

def read_one_row(input):
    with open(input, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            return torch.tensor([float(val) for val in row])

def read_vals(input):
    data = []
    with open(input, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return data