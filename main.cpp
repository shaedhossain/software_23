#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;
#include <cmath>


class Shape {
public:
    virtual double Area() const = 0;
    virtual ~Shape() {}
};

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

double TotalArea(const vector<Shape*>& shapes) {
    double sum = 0;
    for (auto s : shapes) {
        sum += s->Area(); 
    }
    return sum;
}



// Main function to test
int main() {
    vector<Shape*> shapes;
    Shape* s1 = new Rectangle(5, 10);
    Shape* s2 = new Circle(3);
    Shape* s3 = new Elipse(1,1);

    shapes.push_back(s1);
    shapes.push_back(s2);
    shapes.push_back(s3);



    cout << "Rectangle Area: " << s1->Area() << endl; // 50
    cout << "Circle Area: " << s2->Area() << endl; 
    cout << "Elipse Area: " << s3->Area() << endl;  
    
    cout << "Total Area = " << TotalArea(shapes) << endl;

    return 0;
}
