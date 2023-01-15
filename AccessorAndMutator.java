package basics;

public class AccessorAndMutator {
	private String breed;
	private String name;
	private String gender;
	private int age;
// These are instance variables, and they should not have values at the beginning. 

public AccessorAndMutator (String breed, int age) { //This is a constructor
	this.breed = breed;
	this.age = age;
// "This" is used to call higher level variables, mostly instance variables. Another option is creating two unique variables and assigning them.
	}
public AccessorAndMutator (String breed, String name, String gender, int age) {
	this.breed = breed;
	this.name = name;
	this.gender = gender;
	this.age = age; 
	}
// Accessor methods 
// Simpy return the value they want
public String getBreed() {
	return breed;
}
public String getName() {
	return name;
}
public String getGender() {
	return gender;
}
public int getAge() {
	return age;
}

// Mutator methods
// Return a value deviated from the original
public void changeBreed(String breed) { 
	this.breed = breed; //Used for user input
}
public void changeName(String name) {
	this.name = name;
}
public void changeAge(int age) {
	this.age = age;
}

}