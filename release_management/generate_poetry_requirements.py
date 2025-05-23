import subprocess

def add_dependencies_from_requirements(requirements_file, dev=False):
    try:
        with open(requirements_file, 'r') as file:
            for line in file:
                # Clean up the line and ignore empty lines and comments
                package = line.strip()
                if package and not package.startswith('#'):
                    # Prepare the command for adding the package using Poetry
                    command = [
                        'poetry',
                        'add']
                    
                    if dev:
                        command.append('--dev')
                    command.append(package)
                    print(f"Adding package: {package}")
                    # Run the command
                    subprocess.run(
                            command,
                            check=True)
        print(f"Successfully added dependencies from {requirements_file}")
        
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to add a package. Details: {e}")

if __name__ == "__main__":
    # Add main requirements
    add_dependencies_from_requirements(
            r'/requirements.txt')