---
author: "Cong Sun"
author_link: "http://home.ustc.edu.cn/~suncong/"
title: "浅谈Pthread"
date: 2020-05-21T13:24:07+08:00
lastmod: 2020-05-21T13:24:07+08:00
draft: false
description: ""
show_in_homepage: false
description_as_summary: false
license: "MIT license"

tags: ["并行"]
categories: ["技术"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: false
auto_collapse_toc: true
math: false
---

最近由于疫情，一直宅在家里，日子过得非常划水。不过也是趁着这段时间的闲工夫，整理下并行计算一些基本点和常用实现方法。这里就不按照顺序展开了，先介绍下共享存储编程模式中的重要基础--pthread。

# 共享存储
## 简要概述
并行程序与串行程序实现上的重要区别就是执行者的数量不止一个，所以既然不是单打独斗的工作那就需要不同执行者之间进行信息交换，所以一般的并行编程模型(注意这个指的是parallel而不是concurrence)是可以简单的按照不同处理单元(执行者)之间信息交换的方式而进行分类。我们这里主要讨论的是使用共享存储的编程模型，这种编程模型相对于消息传递(MPI)在实现上更为直观，但是实际调优的时候难度也比较大。
在之前的文章中我说明了，在linux中没有像win中标准实现的进程模式，我们的进程和线程都是linux task加了对应不同的flags得到的。进程要求的条件更多，要求有自身独立的堆栈、寄存器和变量空间(code,data,file)；线程相对更为轻量，可以共享全局变量等资源，在一个进程下的不同线程甚至可以访问对方open的文件。
我们对于线程问题也需要注意考虑同步，死锁和缓存一致性。这里我们会讨论前两种问题，缓存一致性问题一般都是通过多处理器总线一致性协议support的，在编程的时候不需要特意关注。
## 主要应用
在真实的使用环境中(我主要的使用环境是科学计算)，线程的创建相对于进程来讲要快上很多，而且及时是最坏情况下，例如我的一个server线程访问numa远端内存的时候，mem-cpu的带宽也是要高于MPI带宽的。但是共享内存的并行方式扩放性不是很好，工业界几乎没有跨节点的共享内存机器(可以通过字典表实现，但是访存延迟过高)，所以在需要大量计算核心跨节点计算的时候，只依靠共享内存的方式是行不通的。我在实际的使用中更倾向于，使用共享内存的方式做单节点上的并行，而节点之前的消息传递则使用消息传递的模式。可以简单的理解为共享存储的并行粒度更细了。

# pthread
有了前面的说明，方便这里简单介绍下pthread。其实早期的时候线程实现对于不同制造商生产的机器是有很大的不同的，但是大家的标准都不能共通，给A机器开发的多线程模式放到B机器上就行不通确实不是一个好主意。所以慢慢的开始有了标准。这里谈的pthread其实就是在POSIX标准中的一部分。POSIX的全称是Potable Operating System Interface,其在Unix-like系统中是通用的。Pthread实际上就是POSIX标准对threads的实现。
## 创建线程
由于pthrea相对low level，所以它的API其实是比较简单的。创建thread API如下：

```
pthread_create(thread,attr,routine,arg)
// thread: 新thread的token id
// attr: 设置thread的某些特性，一般情况下都是NULL
// routine: thread创建之后需要执行的routine
// arg: routine执行的时候需要的参数
```
这里面需要注意的是，pthread在使用上面的API创建threads的时候，是需要同时bind一个routine去给它执行的。在程序的执行过程中，我们以程序入口(main program)进入，然后执行中调用pthread_create创建线程执行一个函数，该函数在执行的时候脱离主线程执行，最后返回值的时候再join回主线程。我用visio画了一个图示(第一次使用+没有鼠标，画的太丑)：
![pthread create标准执行流程](https://img2018.cnblogs.com/blog/1913109/202002/1913109-20200226205241844-939682425.jpg)
值得一提的是，正如之前所讲，pthread是low level的，线程执行函数定义的时候参数列表是void，所以我们不能直接使用arg作为函数变量列表，而是一般使用一个结构体，我们需要在函数执行中，再将之”解包“。而且别忘了函数执行结束之后，将status返回并join到主线程，突出4个字，落叶归根！
除了这个落叶归根的作用之外，如果我们深入的理解下join的过程可以知道其同时也起到了barrier的作用。主线程将会在threadid线程没有join之前block住，可以起到同步的作用。若我们不想使用类似的fork-join模式的时候有要如何操作呢？我们可以使用pthread_detach(threadid)。一旦我们将threadid的线程detach，那么它就相当于离家出走的孩子，不存在落叶归根的问题了(不会join回来)。试想一个使用场景，我们有一个web server来分享某个同化数据，但是不同用户可能需要的数据经纬度网格是不一样的，所以我们在处理的时候往往只是需要一个主线程来监听用户的请求，这个主线程将工作分配给工作线程做一些查找、插值重组过程之后我们会直接将数据返回给用户而不用再告诉主线程。也就是主线程实际上就是起到了大堂经理的作用，真正给我们端茶倒水的还是美丽可人的服务生。这样的模式中，主线程不需要等待工作线程结束，也不需要对工作线程的资源进行回收，实际上还可以节省系统资源。

# 一些多线程问题
从一开始的计算机导论课上我们就知道了程序需要有确定性，同一个程序相同条件的输出应该是确定的。我们玩的可不是薛定谔的计算机。在多线程问题中，我们可以认为不同线程的执行顺序不应当影响到我们最后的结果。
## 同步问题
最简单的例子就是两个线程对于share同一个累加变量的问题(我们可以假设缓存是写直达的，但是这两个线程的寄存器可能不同)。由于变量累加的过程本身不是原子化的，所以其执行过程中需要不止一个指令，这样两个不同线程指令执行的顺序就会影响到最终的结果。一个普遍的想法就是使用编译器提供的原子操作，or设置同步。
### 同步的实现--Lock
同步问题我们知道了，首先可以聊一聊其最直观的实现方式--锁。我们把上面的问题抽象化一些，对于读操作都是ok的，但是对于写操作我们需要给share var加上以临界区来保证同一个写操作时刻，临界区中只能有一个线程在执行。线程一般是先要取得临界区的许可，然后才能进入临界区中搞事情，最后交还许可，退出临界区。
那什么又是lock呢，我们只需要把前面所说的许可换成lock不就可以了。lock有很多实现的方式，linux这里使用的是mutex，由于mutex本身是结构体，所以声名的时候别忘了init一下，like this：
```
#include "pthread.h"

pthread_mutex_t mutex; //这里要保证是global var，要不别的进程咋知道呢
pthread_mutex_init(&mutex, NULL); //init mutex，你好我也好

pthread_mutex_lock(&mutex); //老司机准备进入临界区了，各位准备好

// critical section //

pthread_mutex_unlock(&mutex); //释放锁，有借有还，再借不难
pthread_mutex_destroy(&mutex); // 过河拆桥，锁用不上了

```

## Condition vatiables(cv)
注意，它可不是简历。
cv的含义其实有点偏向于MPI的思想，传递消息。我们可以让threads wait(乖乖站好!)，也可以notify or broadcast线程让他们起来干活。主要的操作有一下三种：
```
phtread_cond_init();

pthread_cond_wait(&theCV,&somelock); //这里sleep线程的同时也将somelock释放掉了，要不其他线程无法取得lock就没办法执行(甚至是叫醒它了）
pthread_cond_signal(&theCV); 
pthread_cond_boardcast(&theCV);
```
需要注意的是，三种thread调度方式需要在一个临界区中。也就是我们一般使用的时候，是需要和上面的lock同时使用的。那么问题来了，为什么一定要这样做呢？
这是因为我们在临界区中处理写share量的时候，需要保证不同线程对其的访问是可控的，否则可能在不同线程中读取到的写share量不一致进而影响CV的工作，因为一般情况下临界区中的写share量就是我们CV工作中的重要判断量。因此，虽然这个条件相对严格，但是是有必要的。

# thread pool
讲到这里，其实我想表达的是pthread本身是很好的实现，它非常灵活，你可以精确的指定每一个线程的routine让它们执行完全不同的任务。但是在享受自由的同时，我们也需要承担应有的责任，这在编程上是十分蛋疼的。所以这里引入一个thread pool的概念来减减压。
如果说我之前所讲的create thread的方式，做个类比，好像是古代打仗的时候临时一个一个拼凑起来的部队，东一个洗一个，那么thread pool则是在runtime中一直存在的一个常备军，waiting for serve。我们可以使用编译器/虚拟机(like java)提供的thread pool创建线程池。这样做的好处是显而易见的，首先我们对于线程的使用是更快的，毕竟不需要每次使用的时候都现场去创建；并且也限制了线程的总数，不会超过系统上线导致server gg。
## thread pool实现
赏析一段源码：
```
/**
 * threadpool.c
 *
 * This file will contain your implementation of a threadpool.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "threadpool.h"

// _threadpool is the internal threadpool structure that is
// cast to type "threadpool" before it given out to callers

typedef struct work_st{
	void (*routine) (void*);
	void * arg;
	struct work_st* next;
} work_t;

typedef struct _threadpool_st {
   // you should fill in this structure with whatever you need
	int num_threads;	//number of active threads
	int qsize;			//number in the queue
	pthread_t *threads;	//pointer to threads
	work_t* qhead;		//queue head pointer
	work_t* qtail;		//queue tail pointer
	pthread_mutex_t qlock;		//lock on the queue list
	pthread_cond_t q_not_empty;	//non empty and empty condidtion vairiables
	pthread_cond_t q_empty;
	int shutdown;
	int dont_accept;
} _threadpool;

/* This function is the work function of the thread */
void* do_work(threadpool p) {
	_threadpool * pool = (_threadpool *) p;
	work_t* cur;	//The q element
	int k;

	while(1) {
		pool->qsize = pool->qsize;
		pthread_mutex_lock(&(pool->qlock));  //get the q lock.


		while( pool->qsize == 0) {	//if the size is 0 then wait.  
			if(pool->shutdown) {
				pthread_mutex_unlock(&(pool->qlock));
				pthread_exit(NULL);
			}
			//wait until the condition says its no emtpy and give up the lock. 
			pthread_mutex_unlock(&(pool->qlock));  //get the qlock.
			pthread_cond_wait(&(pool->q_not_empty),&(pool->qlock));

			//check to see if in shutdown mode.
			if(pool->shutdown) {
				pthread_mutex_unlock(&(pool->qlock));
				pthread_exit(NULL);
			}
		}

		cur = pool->qhead;	//set the cur variable.  

		pool->qsize--;		//decriment the size.  

		if(pool->qsize == 0) {
			pool->qhead = NULL;
			pool->qtail = NULL;
		}
		else {
			pool->qhead = cur->next;
		}

		if(pool->qsize == 0 && ! pool->shutdown) {
			//the q is empty again, now signal that its empty.
			pthread_cond_signal(&(pool->q_empty));
		}
		pthread_mutex_unlock(&(pool->qlock));
		(cur->routine) (cur->arg);   //actually do work.
		free(cur);						//free the work storage.  	
	}
}

threadpool create_threadpool(int num_threads_in_pool) {
  _threadpool *pool;
	int i;

  // sanity check the argument
  if ((num_threads_in_pool <= 0) || (num_threads_in_pool > MAXT_IN_POOL))
    return NULL;

  pool = (_threadpool *) malloc(sizeof(_threadpool));
  if (pool == NULL) {
    fprintf(stderr, "Out of memory creating a new threadpool!\n");
    return NULL;
  }

  pool->threads = (pthread_t*) malloc (sizeof(pthread_t) * num_threads_in_pool);

  if(!pool->threads) {
    fprintf(stderr, "Out of memory creating a new threadpool!\n");
    return NULL;	
  }

  pool->num_threads = num_threads_in_pool; //set up structure members
  pool->qsize = 0;
  pool->qhead = NULL;
  pool->qtail = NULL;
  pool->shutdown = 0;
  pool->dont_accept = 0;

  //initialize mutex and condition variables.  
  if(pthread_mutex_init(&pool->qlock,NULL)) {
    fprintf(stderr, "Mutex initiation error!\n");
	return NULL;
  }
  if(pthread_cond_init(&(pool->q_empty),NULL)) {
    fprintf(stderr, "CV initiation error!\n");	
	return NULL;
  }
  if(pthread_cond_init(&(pool->q_not_empty),NULL)) {
    fprintf(stderr, "CV initiation error!\n");	
	return NULL;
  }

  //make threads

  for (i = 0;i < num_threads_in_pool;i++) {
	  if(pthread_create(&(pool->threads[i]),NULL,do_work,pool)) {
	    fprintf(stderr, "Thread initiation error!\n");	
		return NULL;		
	  }
  }
  return (threadpool) pool;
}


void dispatch(threadpool from_me, dispatch_fn dispatch_to_here,
	      void *arg) {
  _threadpool *pool = (_threadpool *) from_me;
	work_t * cur;
	int k;

	k = pool->qsize;

	//make a work queue element.  
	cur = (work_t*) malloc(sizeof(work_t));
	if(cur == NULL) {
		fprintf(stderr, "Out of memory creating a work struct!\n");
		return;	
	}

	cur->routine = dispatch_to_here;
	cur->arg = arg;
	cur->next = NULL;

	pthread_mutex_lock(&(pool->qlock));

	if(pool->dont_accept) { //Just incase someone is trying to queue more
		free(cur); //work structs.  
		return;
	}
	if(pool->qsize == 0) {
		pool->qhead = cur;  //set to only one
		pool->qtail = cur;
		pthread_cond_signal(&(pool->q_not_empty));  //I am not empty.  
	} else {
		pool->qtail->next = cur;	//add to end;
		pool->qtail = cur;			
	}
	pool->qsize++;
	pthread_mutex_unlock(&(pool->qlock));  //unlock the queue.
}

void destroy_threadpool(threadpool destroyme) {
	_threadpool *pool = (_threadpool *) destroyme;
	void* nothing;
	int i = 0;
/*
	pthread_mutex_lock(&(pool->qlock));
	pool->dont_accept = 1;
	while(pool->qsize != 0) {
		pthread_cond_wait(&(pool->q_empty),&(pool->qlock));  //wait until the q is empty.
	}
	pool->shutdown = 1;  //allow shutdown
	pthread_cond_broadcast(&(pool->q_not_empty));  //allow code to return NULL;
	pthread_mutex_unlock(&(pool->qlock));

	//kill everything.  
	for(;i < pool->num_threads;i++) {
//		pthread_cond_broadcast(&(pool->q_not_empty));
//allowcode to return NULL;/
		pthread_join(pool->threads[i],&nothing);
	}
*/
	free(pool->threads);

	pthread_mutex_destroy(&(pool->qlock));
	pthread_cond_destroy(&(pool->q_empty));
	pthread_cond_destroy(&(pool->q_not_empty));
	return;
}

```
其中，work_st实际上就是task func的一个wrapper，其中包括了work线程需要执行的routine和参数表；在thread_pool结构体的定义中我们也用到了这个struct，结构体中我们声明了需要的lock和thread pool中线程的数据结构queue。这里其实可以理解为这个queue就看做是一个bounded-buffer，我们有生产者调用thread pool，让任务加入到buffer中，也会有worker搞定buffer中的任务成为消费者，我们要做的实际上就是维持这个过程。这里其实可以不使用counter统计bound-buffer中的线程数量，来减少一个临界区的使用(counter++),但是我们需要在消费者和生产者中同时保有tail和head指针用来比较。
在 create_threadpool中，函数声明了必要的资源之后，就是在一个循环中pthread_created线程，对应的routine则是do_work,do_work则会先整理/更新queue数据结构，并且在临界区中当自身pool中activate thread num == 0 && ！(pool not shutdown)的情况下wait对应的signal以备执行，若有需要执行的队列，则会：
```
 (cur->routine) (cur->arg);   //actually do work.
```
执行这个线程所对应的routine。

## semaphore
这个东西可以看作是lock的一个自然延申。也就是一个资源可以同时被多少执行单元使用。我们之前讲到的lock就可以看做是一个binary semaphore。这里就只是简要的谈谈，因为这个东西使用的时候很让人头大，弄不好就会死锁。而且虽然semaphore属于POSIX标准，但是严格来讲的话，它不属于pthread。
使用简谈：
```
#include <semaphore.h>

sem_t sem;
sem_init(&sem);
sem_wait(&sem);

// critical section

sem_post(&sem);
sem_destroy(&sem);
``` 
semaphore可以控制同一个时间有多少thread可以访问临界区，需要注意的就是上面声明--释放的匹配关系不要忘记。为啥不想多谈谈这个呢？你想想，你都有了thread pool，为啥还要玩这个呢？

# 总结
总的来说，pthread这个东西确实功能非常的强大，但是同时实现上也是比较复杂。高级语言如java/python中已经有了monitor这样的工具可以直接使用，不需要我们用这个木头轮子了。在科学计算这一块上，pthread主要是为openmp这样high level一点的编译器特性做支持，日常使用的使用可以用pthread处理一些比较特殊的情况，而主要还是用openmp，mpi，这样更直观的编程模式比较好。
希望疫情可以早点过去，大家的生活重新回到正轨！


<!--more-->
转载请注明出处: http://home.ustc.edu.cn/~suncong/