import random

print("🎯 Welcome to the Guessing Game!")
print("I have picked a number between 1 and 100. Can you guess it?")

# Generate a random number between 1 and 100
secret_number = random.randint(1, 100)

# Max attempts allowed
max_attempts = 7
attempts = 0

while attempts < max_attempts:
    guess = int(input(f"\nAttempt {attempts+1}/{max_attempts} - Enter your guess: "))
    attempts += 1

    if guess < secret_number:
        print("Too low! Try a higher number.")
    elif guess > secret_number:
        print("Too high! Try a lower number.")
    else:
        print(f"🎉 Congrats! You guessed the number {secret_number} in {attempts} tries.")
        break
else:
    print(f"😢 Out of attempts! The number was {secret_number}. Better luck next time!")
