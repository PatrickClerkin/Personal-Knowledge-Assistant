# Design Patterns in Software Engineering

This document covers the most commonly used design patterns in modern software development.

## Creational Patterns

Creational patterns deal with object creation mechanisms, trying to create objects in a manner
suitable to the situation.

### Singleton Pattern

The Singleton pattern ensures a class has only one instance and provides a global point of
access to it. This is useful for resources like database connections, logging services, or
configuration managers where multiple instances would be wasteful or cause conflicts.

### Factory Method

The Factory Method pattern defines an interface for creating an object but lets subclasses
decide which class to instantiate. This promotes loose coupling by eliminating the need to
bind application-specific classes into code.

## Structural Patterns

Structural patterns are concerned with how classes and objects are composed to form larger structures.

### Adapter Pattern

The Adapter pattern allows objects with incompatible interfaces to collaborate. It acts as a
wrapper between two different interfaces, translating calls from one format to another.

### Decorator Pattern

The Decorator pattern lets you attach new behaviours to objects by placing these objects inside
wrapper objects that contain the behaviours. This provides a flexible alternative to subclassing
for extending functionality.

## Behavioural Patterns

Behavioural patterns are concerned with algorithms and the assignment of responsibilities
between objects.

### Strategy Pattern

The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them
interchangeable. This lets the algorithm vary independently from the clients that use it.
The chunking system in this project uses the Strategy pattern extensively.

### Observer Pattern

The Observer pattern defines a one-to-many dependency between objects so that when one object
changes state, all its dependents are notified and updated automatically. This is the foundation
of most event-driven systems and reactive programming frameworks.
