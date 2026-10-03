# 1-OOP-for-ML
Object Oriented Programming in python

---

# Object-Oriented Programming (OOP) for Data Science Workflows

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)

## Overview

As a Machine Learning Engineer, I often see data scientists write monolithic, procedural scripts. While this is fine for quick EDA, transitioning to production environments requires structured, reusable, and maintainable code.

This notebook serves as a foundational guide to **Object-Oriented Programming (OOP)**, demonstrating how software engineering paradigms directly translate into the design of popular ML libraries like `scikit-learn` and `PyTorch`. We transition from basic variable assignments to complex, encapsulated class architectures, culminating in a mock implementation of a Machine Learning Estimator API.

## Core OOP Concepts Demonstrated

This notebook pragmatically covers the pillars of OOP through the analogy of a Banking System before pivoting to a Machine Learning use-case.

### 1. Classes and Instances (The Blueprint vs. The Object)

Instead of defining `car_name`, `car_age`, and `car_mileage` for every single data point, we define a `class`. A class acts as a blueprint, while an instantiated object (like `aks_acc = BankAccount()`) is a concrete realization of that blueprint occupying a distinct memory address.

### 2. State Management (Instance vs. Class Variables)

* **Instance Variables:** Variables prefixed with `self.` (e.g., `self.name`) hold data unique to each specific object.

* **Class Variables:** Variables declared at the class level (e.g., `account_number = 101`). These are shared across *all* instances. In the notebook, this is elegantly used to auto-increment and assign unique account numbers without requiring an external database.

### 3. Dunder (Magic) Methods and Object Interfaces

Dunder (Double Under) methods allow custom objects to interact seamlessly with built-in Python behaviors:

* `__init__(self, ...)`: The constructor. It initializes the state of the object upon instantiation.

* `__repr__(self)`: The official string representation of the object. Crucial for logging and debugging your ML pipelines.

* `__add__(self, other)`: **Operator Overloading / Polymorphism**. This allows us to use the `+` operator to merge two `BankAccount` objects into a joint account, returning a newly instantiated object.

### 4. Encapsulation and Data Hiding

In the notebook, the attribute `amount` is updated to `__amount`. In Python, prefixing an attribute with a double underscore triggers **Name Mangling**. It prevents external entities (or malicious actors, as noted in the "good samaritan" comment) from directly modifying the internal state (`aks_acc.__amount += 100000` will fail). State modification is restricted to controlled methods like `.deposit()` and `.withdrawl()`, enforcing a strict API contract.

### 5. Inheritance and DRY (Don't Repeat Yourself)

Instead of rewriting the banking logic for different account types, we create derived classes: `SavingsAccount(BankAccount)` and `CurrentAccount(BankAccount)`.

* By using the `super().__init__()` function, the child class inherits all parent attributes and methods.

* The child classes then extend functionality (e.g., adding `calculate_interest` to `SavingsAccount`).

## Explaining the Comments (Developer Notes)

Throughout the notebook, there are several inline comments that are vital for understanding the *why* behind the code:

> *"this is to make sure that account numbers always stay unique"* (Inside `BankAccount`)
> **Why this is useful:** It perfectly illustrates the primary use-case for a **Class Variable**. It acts as a global state tracker for the class namespace, ensuring that every new instantiation increments the shared counter, assigning a monotonic ID to each account.

> *"`__repr__` is a dunder method used to display desired output when we type `print(object_name)`"*
> **Why this is useful:** When building custom ML transformers, printing an object usually returns a useless memory address (e.g., `<__main__.BankAccount object at 0x7e...>`). Overriding `__repr__` is a best practice for readability, allowing developers to immediately see the hyperparameter state of a model or the balance of an account.

> *"`y_hat = sigmoid(w.T . X + w0)`"* (Inside `LogisticRegression.predict`)
> **Why this is useful:** This bridges the terminological gap between code and mathematics. It shows that while the OOP wrapper handles the *architecture*, the actual method execution relies on vectorized linear algebra (Weight transpose dot Input Matrix + Bias), which is then passed through an activation function (sigmoid) to output a probability.

## The Machine Learning Connection: Scikit-Learn API Design

The final section of the notebook is the most critical for a data scientist. It demonstrates **Inheritance and API Standardization** in the context of ML.

```python
class LogisticRegression(LinearModel):
    def fit(self, X, y): ...
    def predict(self, X): ...

```

This pseudocode mirrors David Cournapeau's original architecture for `scikit-learn`. By forcing all predictive models to inherit from a base `Estimator` class (like `LinearModel`) and implement standard `.fit()` and `.predict()` methods, we achieve **Polymorphism**.

Because of this OOP design, we can pass *any* model (Logistic Regression, Random Forest, SVM) into a cross-validation loop or pipeline, and the system knows it can blindly call `.fit()` without caring about the underlying algorithmic implementation.
