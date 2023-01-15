package basics;

public class AsseccorAndMutatorTest {
	 public AccessorAndMutator dog; 

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		AccessorAndMutator dog = new AccessorAndMutator("Husky",10);
		System.out.println(dog.getAge());
		System.out.println(dog.getBreed());
		AccessorAndMutator doge = new AccessorAndMutator("Shiba Inu", "dogecoing", "Shitcoin", 3);
		doge.changeAge(4);
		System.out.println(doge.getAge());
	}
}
