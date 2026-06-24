def print_args(args):
    print("-" * 50)
    print(f"{'Argument':<30} | {'Value':<15}")
    print("-" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:<30} | {str(value):<15}")
    print("-" * 50)
