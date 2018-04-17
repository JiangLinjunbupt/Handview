#include "Model.h"
#include "MSRA.h"
#include <iostream>
#include "GL/freeglut.h"
#include "Config.h"
Model* model = nullptr;
double target_position[3*NUM_JOINT];
bool visiable[NUM_JOINT];
msra msr;

#include <iostream>
#include "Viewer.h"
 
//#pragma comment(lib,"freeglutd.lib")

#include "Sampling.h"

#include "Projection.h"

struct CloudPoint {
	//记录思路，如果我们能知道每个点云是由深度图中代表哪个手部部分的像素点得到的，那么我们可以知道这个点云大致对应的手指部分，在后续计算与手摸之间的距离的时候，可以加以利用
	float *cloudpoint;
	int num_cloudpoint;

	CloudPoint() :cloudpoint(nullptr) {};
	~CloudPoint() {
		if (cloudpoint) delete[] cloudpoint;
	}

	void init(cv::Mat depthmat)
	{
		int Num_NotZeroPixel = 0;
		for (int i = 0; i < depthmat.rows; i++)
		{
			for (int j = 0; j < depthmat.cols; j++)
			{
				if (depthmat.at<ushort>(i, j) != 0)
				{
					Num_NotZeroPixel++;
				}
			}
		}

		this->num_cloudpoint = Num_NotZeroPixel;
		this->cloudpoint = new float[num_cloudpoint * 3];
	}

	void DepthMatToCloudPoint(cv::Mat depthmat, double focal, float centerx, float centery)
	{
		int k = 0;
		for (int i = 0; i < depthmat.rows; i++)
		{
			for (int j = 0; j < depthmat.cols; j++)
			{
				if (depthmat.at<ushort>(i, j) != 0)
				{
					this->cloudpoint[k] = (j -centerx) * (-depthmat.at<ushort>(i, j)) / focal;
					this->cloudpoint[k + 1] = -(i - centery) * (-depthmat.at<ushort>(i, j)) / focal;
					this->cloudpoint[k + 2] = - depthmat.at<ushort>(i, j);
					//cout << "the " << k << " point is : x  " << this->cloudpoint[k] << "  y: " << this->cloudpoint[k + 1] << " z: " << this->cloudpoint[k + 2] << endl;
					k = k+3;
				}
			}
		}
	}


};

float Compute_area(cv::Mat input);
float Compute_targetFunction(float area1, cv::Mat input);
void MixShowResult(cv::Mat input1, cv::Mat input2);
cv::Mat generate_depthMap(Pose* pose, float scale, Model* model, Projection *projection);

Config config;
VisData _data;
Control control;
Sample sample(30.0);
CloudPoint _cloudpoint;
Projection *projection = new Projection(240, 320, 23, 28);
cv::Mat groundtruthmat;
const float h = 0.05;
const float step_length = 1;

#pragma region  Keybroad_event(show mesh or not)

void menu(int op) {
 
  switch(op) {
  case 'Q':
  case 'q':
    exit(0);
  }
}
 
/* executed when a regular key is pressed */
void keyboardDown(unsigned char key, int x, int y) {
 
  switch(key) {
  case 'q':
	  config.show_mesh = true;
	  config.show_point = true;
	  config.show_skeleton = false;
	  break;
  case 'w':
	  config.show_mesh = false;
	  config.show_point = true;
	  config.show_skeleton = true;
	  break;

  case  27:   // ESC
    exit(0);
  }
}
 
/* executed when a regular key is released */
void keyboardUp(unsigned char key, int x, int y) {
 
}
 
/* executed when a special key is pressed */
void keyboardSpecialDown(int k, int x, int y) {
 
}
 
/* executed when a special key is released */
void keyboardSpecialUp(int k, int x, int y) {
 
}
#pragma endregion  Keybroad_event(show mesh or not)


/* reshaped window */
void reshape(int width, int height) {
 
  GLfloat fieldOfView = 90.0f;
  glViewport (0, 0, (GLsizei) width, (GLsizei) height);
 
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fieldOfView, (GLfloat) width/(GLfloat) height, 0.1, 500.0);
 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}
 
/* executed when button 'button' is put into state 'state' at screen position ('x', 'y') */
void mouseClick(int button, int state, int x, int y) {
	control.mouse_click = 1;
	control.x = x;
	control.y = y;
}
 
/* executed when the mouse moves to position ('x', 'y') */
void logo(){
  glRasterPos2i(100, 100);
  glColor3d(0.0, 0.0, 1.0);
  const unsigned char kurff0[] = "kurff";
  glutBitmapString(GLUT_BITMAP_HELVETICA_18, kurff0);
   glRasterPos2i(-100, 100);
  glColor3d(0.0, 1.0, 0.0);  //(red ,green ,blue)
  const unsigned char kurff1[] = "kurff";
  glutBitmapString(GLUT_BITMAP_HELVETICA_18, kurff1);
     glRasterPos2i(100, -100);
  glColor3d(1.0, 0.0, 0.0);
  const unsigned char kurff2[] = "kurff";
  glutBitmapString(GLUT_BITMAP_HELVETICA_18, kurff2);
   glRasterPos2i(-100, -100);
  glColor3d(1.0, 1.0, 0);
  const unsigned char kurff3[] = "kurff";
  glutBitmapString(GLUT_BITMAP_HELVETICA_18, kurff3);
}
 
/* render the scene */
void draw() {
 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  gluPerspective(180, 1.5, -1000, 1000); 
  glLoadIdentity();
  control.gx = model->get_global_position().x;
  control.gy = model->get_global_position().y;
  control.gz = model->get_global_position().z;
  double r = 200;
  double x = r*sin(control.roty)*cos(control.rotx);
  double y = r*sin(control.roty)*sin(control.rotx);
  double z = r*cos(control.roty);
  //cout<< x <<" "<< y <<" " << z<<endl;
  gluLookAt(x+control.gx,y+control.gy,z+control.gz,control.gx,control.gy,control.gz,0.0, 1.0, 0.0);//个人理解最开始是看向-z的，之后的角度是在global中心上叠加的，所以要加

  logo();
  /* render the scene here */
  // glColor3d(1.0,1.0,1.0);
  if(config.show_point){
	  glPointSize(2);
	  glBegin(GL_POINTS);
      glColor3d(1.0,0.0,0.0);
	  for(int i = 0; i < model->vertices_update_.rows(); i ++ ){		  
		  glVertex3d(model->vertices_update_(i,0),model->vertices_update_(i,1),model->vertices_update_(i,2));
		  //cout<< model->vertices_update_(i,0)<< " " << model->vertices_update_(i,1) <<" "<< model->vertices_update_(i,2)<<endl;
	  }
	  glEnd();
  }

  if (config.show_mesh) {
	  if (_data.indices == nullptr) return;
	  if (_data.vertices == nullptr) return;
	  glColor3d(0.0, 0.0, 1.0);
	  glEnableClientState(GL_VERTEX_ARRAY);
	  glVertexPointer(3, GL_FLOAT, 0, _data.vertices);
	  glEnableClientState(GL_COLOR_ARRAY);
	  glColorPointer(3, GL_FLOAT, 0, _data.colors);
	  //glDrawElements(GL_TRIANGLE_STRIP, 12, GL_UNSIGNED_BYTE, indices);
	  glDrawElements(GL_TRIANGLES, 3 * _data.num_face, GL_UNSIGNED_INT, _data.indices);

	  // deactivate vertex arrays after drawing
	  glDisableClientState(GL_VERTEX_ARRAY);

  }
  //glEnable(GL_LIGHTING);
  if(config.show_skeleton){
	  for(int i = 0; i < _data.joints.rows(); i++ ){
		  //画点开始
		  glColor3f(1.0,0.0,0.0); 
		  glPushMatrix();
		  glTranslatef(_data.joints(i,0), _data.joints(i,1), _data.joints(i,2));
		  glutSolidSphere(5, 31, 10);
		  glPopMatrix();
		  //画点结束，使用push和popmatrix是因为保证每个关节点的偏移都是相对于全局坐标中心点做的变换。

		  //画线开始
		  if(i!=0){
			  glLineWidth(5);
			  glColor3f(0.0,1.0,0); 
			  glBegin(GL_LINES);
			  int ii = _data.joints(i,3);
			  glVertex3f(_data.joints(ii,0), _data.joints(ii,1), _data.joints(ii,2));
			  glVertex3f(_data.joints(i,0), _data.joints(i,1), _data.joints(i,2));
			  glEnd();		
		  }

		  //画线结束
	  }
  }

  glPointSize(5);
  glBegin(GL_POINTS);
  glColor3d(1.0, 0.0, 0.0);
  for (int i = 0; i < _cloudpoint.num_cloudpoint*3; i = i+3)
  {
	  glVertex3d(_cloudpoint.cloudpoint[i], _cloudpoint.cloudpoint[i+1], _cloudpoint.cloudpoint[i+2]);
  }
  glEnd();


  glLineWidth(5);
  glColor3f(1.0, 0.0, 0);
  glBegin(GL_LINES);
  glVertex3f(0, 0,0);
  glVertex3f(100, 0, 0);
  glEnd();

  glLineWidth(5);
  glColor3f(0.0, 1.0, 0);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(0,100, 0);
  glEnd();

  glLineWidth(5);
  glColor3f(0.0, 0.0, 1.0);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, 100);
  glEnd();

  glFlush();
  glutSwapBuffers();
}


void mouseMotion(int x, int y) {
	control.rotx = (x - control.x)*0.05;
	control.roty = (y - control.y)*0.05;

    //cout<< control.rotx <<" " << control.roty << endl;
	glutPostRedisplay();
}

/* executed when program is idle */
void idle() { 
	//for(int i = 0; i < msr.nframe_; i ++ ){
	//	msr.msra_to_model(target_position,i);	
	//	model->set_target_position(target_position,visiable);
	//	for(int j = 0; j < NUM_JOINT*3; j ++ ){
	//	    std::cout<< target_position[j] << " "; 
	//	}
	//	model->inverse_kinematic();
	//	model->forward_kinematic();
	//	model->compute_mesh();
	//	data.set_vertices(model->vertices_update_);
	//	draw();
	//}
	Pose pose;
	pose.x = 0; pose.y = 0; pose.z = -200;

	//sample.select_one(model);           //用于随机选择一个关节点，进行该关节点变换范围内的随机变换，并且后续进行生成图像等一系列操作
	model->forward_kinematic();
	model->compute_mesh();
	projection->compute_current_orientation(model);
	projection->project_3d_to_2d(model);
	_data.set_vertices(model->vertices_update_);
	_data.set_skeleton(model);
	glutPostRedisplay();
}
 
/* initialize OpenGL settings */
void initGL(int width, int height) {
 
  reshape(width, height);
 
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(1.0f);
 
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
}


void main(int argc, char** argv){
    //Model model("HandBase.bvh");
	model = new Model(".\\model\\HandBase.bvh");
    model->init();

	Pose pose(0,0,0);
	pose.x = 0; pose.y = -0; pose.z = -20;
	model->set_one_rotation(pose,21);  
	model->load_faces(".\\model\\handfaces.txt");
	model->load_vertices(".\\model\\handverts.txt");
	model->load_weight(".\\model\\weight.txt");

	Configuration* pconfig = Global();
	pconfig->LoadConfiguration("configuration.txt");
	 
	pose.x = 0; pose.y = 0; pose.z = -400;   //这里z负方向上越大，深度图中的手看起来越小
	model->set_global_position(pose);
	model->set_global_position_center(pose);

	//cv::Mat img = cv::imread("groundtruth.png", CV_LOAD_IMAGE_UNCHANGED);  //这里采用CV_LOAD_IMAGE_UNCHANGED这个模式才可以真确的读入，不然读入都是不正确的，可能和存储的深度值是16位有关系。
	//cv::Mat show_depth = cv::Mat::zeros(240, 320, CV_8UC1);
	//for (int i = 0; i < img.rows; i++)
	//{
	//	for (int j = 0; j < img.cols; j++) {
	//		show_depth.at<uchar>(i, j) = img.at<ushort>(i, j) % 255;
	//	}
	//}
	//cv::imshow("kkk", show_depth);
 //   cv::waitKey(100);

	model->Load_groundtruth_pose("groundtruth.txt", model);
	groundtruthmat = generate_depthMap(model->given_pose, model->given_scale, model, projection);

	_cloudpoint.init(groundtruthmat);
	_cloudpoint.DepthMatToCloudPoint(groundtruthmat, 241.3, 160, 120);
	cout << _cloudpoint.num_cloudpoint << endl;

	//float groundarea = Compute_area(groundtruthmat);

	//srand(time(NULL));
	//model->given_scale = float(10 * rand() / float(RAND_MAX + 1));

	//float difference = 0;
	//do
	//{
	//	float gradient = 0;
	//	for (int i = 0; i < 23; i++)
	//	{
	//		model->set_joint_scale(model->given_scale + h, i);
	//	}

	//	cv::Mat generatedmat1 = generate_depthMap(model->given_pose, model->given_scale, model, projection);


	//	float gradient1 = Compute_targetFunction(groundarea, generatedmat1);


	//	for (int i = 0; i < 23; i++)
	//	{
	//		if (model->given_scale > h)
	//		{
	//			model->set_joint_scale(model->given_scale - h, i);
	//		}
	//		else
	//		{
	//			model->set_joint_scale(model->given_scale, i);
	//		}
	//	}

	//	cv::Mat generatedmat2 = generate_depthMap(model->given_pose, model->given_scale, model, projection);

	//	float gradient2 = Compute_targetFunction(groundarea, generatedmat2);
	//	gradient = gradient1- gradient2;
	//	cout << gradient1 << "  ";
	//	gradient = gradient / (2 * h);
	//	cout << gradient2 << "  " << gradient << "  ";
	//	if (gradient != 0)
	//	{
	//		//gradient = gradient / abs(gradient);
	//		model->given_scale = model->given_scale - step_length*gradient;


	//		for (int i = 0; i < 23; i++)
	//		{
	//			model->set_joint_scale(model->given_scale, i);
	//		}

	//		cv::Mat generatedmat = generate_depthMap(model->given_pose, model->given_scale, model, projection);

	//		difference = Compute_targetFunction(groundarea, generatedmat);
	//		cout << difference << "  " << model->given_scale << endl;
	//		MixShowResult(groundtruthmat, generatedmat);
	//		cv::waitKey(50);
	//	}
	//	else
	//	{
	//		cout << "gradient is 0"<<"  ";
	//		cv::Mat generatedmat = generate_depthMap(model->given_pose, model->given_scale, model, projection);

	//		difference = Compute_targetFunction(groundarea, generatedmat);
	//		cout << difference << "  " << model->given_scale << endl;
	//		MixShowResult(groundtruthmat, generatedmat);
	//		cv::waitKey(50);
	//	}
	//	
	//} while (difference >0.001);


	//生成真实值begin ，并且保存真实值为txt文件和png图片文件
	/*model->Random_given_poseAndscale(model);
	model->Save_given_params("groundtruth.txt");
	cv::Mat grounthmat = generate_depthMap(model->given_pose, model->given_scale, model, projection);
	cv::imwrite("groundtruth.png", grounthmat);*/
	//生成真实值end



	//model->forward_kinematic();
	//model->compute_mesh();

	//projection->set_color_index(model);
	//projection->compute_current_orientation(model);
	//projection->project_3d_to_2d(model);


	//用于opengl显示
	_data.init(model->vertices_.rows(), model->faces_.rows());
	_data.set(model->vertices_update_, model->faces_);
	_data.set_color(model->weight_);
	_data.set_skeleton(model);
	//model->compute_mesh();
	//double target_position[3*NUM_JOINT];
	//memset(target_position,0,sizeof(double)*NUM_JOINT*3);
	//bool visiable[NUM_JOINT];
	//memset(visiable,0,sizeof(bool)*NUM_JOINT);

	//ofstream f;
	//f.open("result.txt",ios::out);
	//for(int i= 0 ; i< model->vertices_.rows(); i ++ ){
	//	for(int j = 0; j < model->vertices_.cols(); j ++ ){
	//		f<<model->vertices_(i,j)<<" ";
	//	}
	//	f<<endl;
	//
	//}
	//f.close();

	//memset(visiable,1,sizeof(bool)*NUM_JOINT);
	//visiable[0] = 0; visiable[NUM_JOINT-1] = 0;
	//msr.read_joint_msra("joint.txt");
	//msr.init();
	//msr.set_visiable(visiable);


	//msra msr;
	//msr.read_joint_msra("joint.txt");
	//msr.init();
	//msr.set_visiable(visiable);
	//for(int i = 0; i < msr.nframe_; i ++ ){
	//	msr.msra_to_model(target_position,i);	
	//	model.set_target_position(target_position,visiable);
	//	for(int j = 0; j < NUM_JOINT*3; j ++ ){
	//	    std::cout<< target_position[j] << " "; 
	//	}
	//	model.inverse_kinematic();
	//}



	//glut简单教程，以下的函数都有提到：http://www.cnblogs.com/yangxi/archive/2011/09/16/2178479.html
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Interactron");

	// register glut call backs
	glutKeyboardFunc(keyboardDown);
	glutKeyboardUpFunc(keyboardUp);
	glutSpecialFunc(keyboardSpecialDown);
	glutSpecialUpFunc(keyboardSpecialUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutReshapeFunc(reshape);  //当使用鼠标改变opengl显示窗口时，被调用的函数，保证不变形
	glutDisplayFunc(draw);
	//glutIdleFunc(idle);
	glutIgnoreKeyRepeat(true); // ignore keys held down

							   // create a sub menu 
	int subMenu = glutCreateMenu(menu);
	glutAddMenuEntry("Do nothing", 0);
	glutAddMenuEntry("Really Quit", 'q');

	// create main "right click" menu
	glutCreateMenu(menu);
	glutAddSubMenu("Sub Menu", subMenu);
	glutAddMenuEntry("Quit", 'q');
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	initGL(800, 600);
	glutMainLoop();






	//model->save("position.txt");
	//model->save2("init_position.txt");
	//model->save_trans("trans.txt");
	//model->save_local("local.txt");
	//model->save_global("global.txt");
	system("PAUSE");
}


float Compute_area(cv::Mat input)
{
	int HandPixelCount = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<ushort>(i, j) != 0)
				HandPixelCount++;
		}
	}

	return ((float)HandPixelCount)/(input.rows*input.cols);
}

float Compute_targetFunction(float area1, cv::Mat input)
{
	float area2 = Compute_area(input);
	return abs(area1 - area2);
}

void MixShowResult(cv::Mat input1, cv::Mat input2)
{
	int height = input2.rows;
	int width = input2.cols;
	cv::Mat colored_input1 = cv::Mat::zeros(height, width, CV_8UC3);
	cv::Mat colored_input2 = cv::Mat::zeros(height, width, CV_8UC3);
	cv::Mat dst;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (input1.at<ushort>(i, j) != 0)
			{
				colored_input1.at < cv::Vec3b>(i, j)[0] = 0;
				colored_input1.at < cv::Vec3b>(i, j)[1] = 0;
				colored_input1.at < cv::Vec3b>(i, j)[2] = 255;
			}
			else
			{

				colored_input1.at < cv::Vec3b>(i, j)[0] = 255;
				colored_input1.at < cv::Vec3b>(i, j)[1] = 255;
				colored_input1.at < cv::Vec3b>(i, j)[2] = 255;

			}

			if (input2.at<ushort>(i, j) != 0)
			{
				colored_input2.at < cv::Vec3b>(i, j)[0] = 0;
				colored_input2.at < cv::Vec3b>(i, j)[1] = 255;
				colored_input2.at < cv::Vec3b>(i, j)[2] = 0;
			}
			else
			{

				colored_input2.at < cv::Vec3b>(i, j)[0] = 255;
				colored_input2.at < cv::Vec3b>(i, j)[1] = 255;
				colored_input2.at < cv::Vec3b>(i, j)[2] = 255;

			}

		}
	}

	cv::addWeighted(colored_input1, 0.5, colored_input2, 0.5, 0.0, dst);
	cv::imshow("Mixed Result", dst);

}

cv::Mat generate_depthMap(Pose* pose, float scale, Model* model, Projection *projection)
{
	cv::Mat generated_mat = cv::Mat::zeros(240, 320, CV_16UC1);;
	for (int i = 1; i < 23; i++)
	{
		model->set_one_rotation(pose[i], i);
		model->set_joint_scale(scale, i);
		
	}

	model->forward_kinematic();
	model->compute_mesh();

	projection->set_color_index(model);
	projection->compute_current_orientation(model);
	projection->project_3d_to_2d_(model,generated_mat);
	return generated_mat;
}

