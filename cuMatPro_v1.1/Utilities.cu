#include "Utilities.cu.h"

map<string, int> Utilities::groupCounts;
map<string, vector<string> > Utilities::groupedImages;

Utilities::Utilities()
{
}

void Utilities::DisplayMat(Mat &image, string s)
{
	cout << s << " of size " << image.rows << ", " << image.cols << endl;
	if (s.size() == 0) {
		namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
		imshow("Display window", image); // Show our image inside it.
	}
	else {
		namedWindow(s, WINDOW_AUTOSIZE); // Create a window for display.
		imshow(s, image); // Show our image inside it.
	}
}

vector<string> Utilities::GetAllFiles(string imageCategory)
{	
	if (groupCounts.size() == 0) {
		cout << "Generating file list..." << endl;
		ifstream inf("rfiles/pointerList.list1");
		char a[80];
		while (!inf.eof()) {
			inf >> a;
			string cat(a);
			if (cat.size() == 0) {
				break;
			}
			inf >> a;
			groupCounts[cat] = atoi(a);
			
			ifstream catFileStream("rfiles/" + cat + ".list2");
			groupedImages[cat] = vector<string>();
			char b[1024];
			while (!catFileStream.eof()) {
				catFileStream >> b;
				if (strlen(b) == 0) {
					break;
				}
				groupedImages[cat].push_back(string(b));
			}
			catFileStream.close();
		}
		inf.close();
	}
	
	return groupedImages[imageCategory];	
}

void Utilities::prettyStart(string s)
{
	cout << endl << "***************************************" << endl;
	cout << "\t" << s << endl;
	cout << "***************************************" << endl << endl;
}
void Utilities::prettyEnd(string s)
{
	cout << endl << "=======================================" << endl;
	cout << "\t" << s << endl;
	cout << "=======================================" << endl << endl;
}

Utilities::~Utilities()
{
}
