import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from test import test_statement
        print("\n\n")
        test_statement(sys.argv[1])
    else:
        print("Please provide a string as an argument.")