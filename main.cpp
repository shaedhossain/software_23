#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;
#include <cmath>


// Abstract class (Interface-like)
class Shape {
public:
    // Pure virtual function (abstract method)
    virtual double Area() const = 0;

    // Always good to have a virtual destructor in base classes
    virtual ~Shape() {}
};

// Rectangle class
class Rectangle : public Shape {
public:
    double width;
    double height;

    Rectangle(double w, double h){
        width=w;
        height=h;
    }

    double Area() const override {
        return width * height;
    }
};

// Circle class
class Circle : public Shape {
public:
    double radius;

    Circle(double r){
        radius=r;
    }

    double Area() const override {
        return radius * radius * 3.1416;
    }
};

class Elipse : public Shape {
public:
    double r1, r2;

    Elipse(double a, double b){
        r1=a;
        r2=b;
    }

    double Area() const override {
        return r1 * r2 * 3.1416;
    }
};



// Main function to test
int main() {
    Shape* s1 = new Rectangle(5, 10);
    Shape* s2 = new Circle(3);
    Shape* s3 = new Elipse(1,1);


    cout << "Rectangle Area: " << s1->Area() << endl; // 50
    cout << "Circle Area: " << s2->Area() << endl; 
    cout << "Elipse Area: " << s3->Area() << endl;    // ~28.27

    // Clean up
    delete s1;
    delete s2;

    return 0;
}
